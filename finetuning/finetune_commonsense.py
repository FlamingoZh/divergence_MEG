import argparse
import os
import datasets
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.deepspeed import HfDeepSpeedConfig
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.lamb import FusedLamb
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import deepspeed
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import logging
from tqdm import tqdm
from typing import List, Tuple, Optional
import copy
import wandb
import json
import random
from sklearn.model_selection import train_test_split, KFold


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_columns = {
    "social_i_qa": {"question": ["context", "question"], "options": ["answerA", "answerB", "answerC"], "answer": "label"},
    "piqa": {"question": ["goal"], "options": ["sol1", "sol2"], "answer": "label"},
    "nightingal3/fig-qa": {"question": ["startphrase"], "options": ["ending1", "ending2"], "answer": "labels"}
}

def format_dataset_for_mc(dataset, dataset_name) -> Tuple[List, List, List]:
    questions = []
    options = []
    answers = []

    for datapoint in dataset:
        question = " ".join([datapoint[col] for col in dataset_columns[dataset_name]["question"]])
        option = [datapoint[col] for col in dataset_columns[dataset_name]["options"]]
        answer = datapoint[dataset_columns[dataset_name]["answer"]]

        questions.append(question)
        options.append(option)
        answers.append(answer)

    return questions, options, answers

def process_book_file(book_path: str, tokenizer: GPT2Tokenizer, max_length: Optional[int] = None) -> List[str]:
    lines = []
    max_length = max_length if max_length is not None else tokenizer.model_max_length
    with open(book_path, "r") as f:
        for line in f:
            line = line.strip()
            len_line = len(tokenizer.encode(line))
            if len_line > 1 and len_line < max_length:
                lines.append(line)
            
            if len(line) > max_length:
                print("Excluding line")
            
    return lines

class MCDataset(Dataset):
    def __init__(self, questions, options, answers, tokenizer, is_one_indexed=True):
        self.questions = questions
        self.options = options
        self.answers = answers
        self.num_options = len(options[0])
        self.is_one_indexed = is_one_indexed

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        options = self.options[idx]
        answer = self.answers[idx]

        inputs = [self.tokenizer.encode_plus(question, option, add_special_tokens=True, max_length=1024, padding="max_length") for option in options]

        return {
            "input_ids": torch.stack([torch.tensor(inp["input_ids"]) for inp in inputs]),
            "attention_mask": torch.stack([torch.tensor(inp["attention_mask"]) for inp in inputs]),
            "labels": self.label_to_one_hot(int(answer))
        }
    
    def label_to_one_hot(self, label):
        one_hot = torch.zeros(self.num_options)
        if self.is_one_indexed:
            one_hot[label - 1] = 1
        else:
            one_hot[label] = 1
        return one_hot


class LMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors='pt'
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }

class CombinedDataset(Dataset):
    def __init__(self, mcq_dataset, lm_dataset, lm_p=0.5):
        self.mcq_dataset = mcq_dataset 
        self.lm_dataset = lm_dataset 
        self.lm_p = lm_p
        
        # Ensure that the datasets are of the same length for zipping
        self.length = max(len(mcq_dataset), len(lm_dataset))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Decide which dataset to sample from based on the probability p
        if self.lm_p == 0:
            return {"type": "mcq", **self.mcq_dataset[idx]}
        elif self.lm_p == 1:
            return {"type": "lm", **self.lm_dataset[idx]}
        else:
            if random.random() < self.lm_p:
                if idx >= len(self.mcq_dataset):
                    # If you've exhausted the MCQ dataset, then loop around
                    idx = idx % len(self.mcq_dataset)
                return {"type": "mcq", **self.mcq_dataset[idx]}
            else:
                if idx >= len(self.lm_dataset):
                    # If you've exhausted the LM dataset, then loop around
                    idx = idx % len(self.lm_dataset)
                return {"type": "lm", **self.lm_dataset[idx]}

def validate_mcq(model, valid_dataloader_mcq, using_deepspeed=True) -> Tuple[float, int]:
    model.eval()
    total_loss_valid_mcq = 0
    num_correct_val = 0
    with torch.no_grad():
        for batch in tqdm(valid_dataloader_mcq):
            input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
            if using_deepspeed:
                input_ids = input_ids.to(model.local_rank)
                attention_mask = attention_mask.to(model.local_rank)
                labels = labels.to(model.local_rank)
            else:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            logits = outputs.logits # (1, num_options, seq_len, vocab_size)
            logits = torch.gather(logits, dim=3, index=input_ids.unsqueeze(3)).squeeze(3) # (1, num_options, seq_len)
            logits = logits * attention_mask
            logits_sum = torch.sum(logits, dim=2) # (1, num_options)
            loss = torch.nn.functional.cross_entropy(logits_sum, labels)
            total_loss_valid_mcq += loss.item()

            # accuracy 
            _, predicted = torch.max(logits_sum, 1)
            if predicted == torch.argmax(labels):
                num_correct_val += 1

    return total_loss_valid_mcq, num_correct_val

def validate_lm(model, valid_dataloader_lm, using_deepspeed=True, local_rank=None) -> float:
    model.eval()
    with torch.no_grad():
        total_loss_valid_lm = 0
        for batch in tqdm(valid_dataloader_lm):
            input_ids, attention_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
            if using_deepspeed:
                try:
                    input_ids = input_ids.to(model.local_rank)
                    attention_mask = attention_mask.to(model.local_rank)
                    labels = labels.to(model.local_rank)
                except:
                    input_ids = input_ids.to(local_rank)
                    attention_mask = attention_mask.to(local_rank)
                    labels = labels.to(local_rank)
            else:
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                labels = labels.to(DEVICE)
            

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss_valid_lm += loss.item()

    return total_loss_valid_lm

def train_all_options(model, tokenizer, train_set, valid_set, sel_datasets, lm_train_set=None, lm_valid_set=None, lm_test_set=None, early_stopping=0, lm_task_p=0.0, fp16=False, using_deepspeed=False, dsconfig=None, using_wandb=False, output_dir="./out", local_rank=-1, n_epochs=10, convert_to_fp32_now=False, log_steps_frequency=1000, fold=0):
    if len(sel_datasets) == 1:
        train_qs, train_options, train_answers = format_dataset_for_mc(train_set[0], sel_datasets[0])
        valid_qs, valid_options, valid_answers = format_dataset_for_mc(valid_set[0], sel_datasets[0])
    else:
        raise NotImplementedError
    
    if lm_task_p > 0:
        assert lm_train_set is not None and lm_valid_set is not None

    if fp16:
        scaler = GradScaler()

    is_one_indexed = True if sel_datasets[0] == "social_i_qa" else False
    train_set = MCDataset(train_qs, train_options, train_answers, tokenizer, is_one_indexed=is_one_indexed)
    valid_set = MCDataset(valid_qs, valid_options, valid_answers, tokenizer, is_one_indexed=is_one_indexed)

    lm_train_set = LMDataset(lm_train_set["text"], tokenizer, tokenizer.model_max_length)
    lm_valid_set = LMDataset(lm_valid_set["text"], tokenizer, tokenizer.model_max_length)
    lm_test_set = LMDataset(lm_test_set["text"], tokenizer, tokenizer.model_max_length)

    if lm_task_p == 0:
        lm_train_set = []
    elif lm_task_p == 1:
        train_set = []

    combined_dataset_train = CombinedDataset(train_set, lm_train_set, lm_p=lm_task_p)

    combined_train_dataloader = DataLoader(combined_dataset_train, batch_size=1, shuffle=True)

    valid_dataloader_mcq = DataLoader(valid_set, batch_size=1, shuffle=False)

    valid_dataloader_lm = DataLoader(lm_valid_set, batch_size=1, shuffle=False)
    
    test_dataloader_lm = DataLoader(lm_test_set, batch_size=1, shuffle=False)

    if using_deepspeed:
        if dsconfig is None:
            dsconfig = dict(json.load(open("ds_config.json", "r")))

        model, optimizer, _, _ = deepspeed.initialize(model=model, config_params=dsconfig)
    else:    
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    num_epochs = n_epochs if early_stopping == 0 else 100
    best_val_loss = float("inf")
    patience = early_stopping

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch}")
        logging.info("Training")

        model.train()
        total_epoch_loss_mcq = 0
        total_epoch_loss_lm = 0
        running_loss_mcq = 0
        running_loss_lm = 0
        num_correct = 0
        total_epoch_num_correct = 0
        num_mcq_questions = 0
        num_lm_questions = 0
        total_lm_questions = 0
        total_mcq_questions = 0
        #for i, mc_batch in enumerate(tqdm(train_dataloader)):
        for i, batch in enumerate(tqdm(combined_train_dataloader)):
            optimizer.zero_grad()
            task_type = batch["type"]

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            if using_deepspeed:
                input_ids = input_ids.to(model.local_rank)
                attention_mask = attention_mask.to(model.local_rank)
                labels = labels.to(model.local_rank)
            else:
                input_ids = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                labels = labels.to(DEVICE)

            if fp16:
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

                    if task_type == "mcq":
                        logits = outputs.logits # (1, num_options, seq_len, vocab_size)
                        logits = torch.gather(logits, dim=3, index=input_ids.unsqueeze(3)).squeeze(3) # (1, num_options, seq_len)
                        logits = logits * attention_mask
                        logits_sum = torch.sum(logits, dim=2) # (1, num_options)
                        loss = torch.nn.functional.cross_entropy(logits_sum, labels)
                        running_loss_mcq += loss.detach().item()
                        total_epoch_loss_mcq += loss.detach().item()

                        num_mcq_questions += 1
                        total_mcq_questions += 1
            
                    else:
                        loss = outputs.loss
                        running_loss_lm += loss.detach().item()
                        total_epoch_loss_lm += loss.detach().item()

                        num_lm_questions += 1
                        total_lm_quesions += 1
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

                if task_type == "mcq":
                    logits = outputs.logits # (1, num_options, seq_len, vocab_size)
                    logits = torch.gather(logits, dim=3, index=input_ids.unsqueeze(3)).squeeze(3) # (1, num_options, seq_len)
                    logits = logits * attention_mask
                    logits_sum = torch.sum(logits, dim=2) # (1, num_options)
                    loss = torch.nn.functional.cross_entropy(logits_sum, labels)
                    running_loss_mcq += loss.detach().item()
                    total_epoch_loss_mcq += loss.detach().item()
                    num_mcq_questions += 1
                    total_mcq_questions += 1
                else:
                    loss = outputs.loss
                    running_loss_lm += loss.detach().item()
                    total_epoch_loss_lm += loss.detach().item()

                    num_lm_questions += 1
                    total_lm_questions += 1
            # accuracy 

            if task_type == "mcq":
                _, predicted = torch.max(logits_sum, 1)
                if predicted == torch.argmax(labels):
                    num_correct += 1
                    total_epoch_num_correct += 1                

            if using_deepspeed:
                model.backward(loss)
                model.step()
            elif fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # delete tensors in case they're causing memory issues
            del input_ids
            del attention_mask
            del labels
            torch.cuda.empty_cache()

            if (i % log_steps_frequency == 0) and (i != 0): 
                if lm_task_p != 1:
                    logging.info(f"Avg train loss (MCQ): {running_loss_mcq / num_mcq_questions}")
                if lm_task_p != 0:
                    logging.info(f"Avg train loss (LM): {running_loss_lm / num_lm_questions}")

                results_train = {}
                if lm_task_p != 1:
                    results_train.update({"mcq_loss": running_loss_mcq / log_steps_frequency, "mcq_accuracy": num_correct / num_mcq_questions})
                elif lm_task_p != 0:
                    results_train.update({"lm_loss": running_loss_lm / log_steps_frequency})

                running_loss_mcq = 0
                running_loss_lm = 0
                num_mcq_questions = 0
                num_correct = 0
                num_lm_questions = 0

                logging.info(f"Evaluating at step {i}")
                model.eval()

                logging.info("Evaluating the MCQ task")
                if not using_deepspeed or (using_deepspeed and local_rank == 0):
                    total_loss_valid_mcq, num_correct_val = validate_mcq(model, valid_dataloader_mcq, using_deepspeed=using_deepspeed)
                    
                    logging.info(f"Avg valid loss (MCQ): {total_loss_valid_mcq / len(valid_dataloader_mcq)}")
                    if using_wandb and (not using_deepspeed or (using_deepspeed and local_rank == 0)):
                        if lm_task_p != 1:
                            wandb.log({
                                "step": i,
                                "train_mcq_loss": results_train["mcq_loss"],
                                "train_mcq_accuracy": results_train["mcq_accuracy"],
                                "valid_mcq_loss": total_loss_valid_mcq / len(valid_dataloader_mcq),
                                "valid_mcq_accuracy": num_correct_val / len(valid_dataloader_mcq)
                            })
                        else:
                            wandb.log({
                                "step": i,
                                "valid_mcq_loss": total_loss_valid_mcq / len(valid_dataloader_mcq),
                                "valid_mcq_accuracy": num_correct_val / len(valid_dataloader_mcq)
                            })

                logging.info("Evaluating the LM task")
                if not using_deepspeed or (using_deepspeed and local_rank == 0):
                    total_loss_lm = validate_lm(model, valid_dataloader_lm, using_deepspeed=using_deepspeed)
                    logging.info(f"Avg valid loss (LM): {total_loss_lm / len(valid_dataloader_lm)}")
                    if using_wandb:
                        if lm_task_p != 0:
                            wandb.log({
                                "step": i,
                                "train_lm_loss": results_train["lm_loss"],
                                "valid_lm_loss": total_loss_lm / len(valid_dataloader_lm)
                            })
                        else:
                            wandb.log({
                                "step": i,
                                "valid_lm_loss": total_loss_lm / len(valid_dataloader_lm)
                            })

                model.train()

        # End of epoch
        logging.info("End of epoch, evaluating")

        model.eval()

        logging.info("Evaluating the MCQ task")
        if not using_deepspeed or (using_deepspeed and local_rank == 0):
            total_loss_valid_mcq, num_correct_val = validate_mcq(model, valid_dataloader_mcq, using_deepspeed=using_deepspeed)

            logging.info(f"Avg valid loss: {total_loss_valid_mcq / len(valid_dataloader_mcq)}")
            if using_wandb:
                if lm_task_p != 1:
                    wandb.log({
                        "epoch": epoch,
                        "train_mcq_loss": total_epoch_loss_mcq / total_mcq_questions,
                        "train_mcq_accuracy": total_epoch_num_correct / total_mcq_questions,
                        "valid_mcq_loss": total_loss_valid_mcq / len(valid_dataloader_mcq),
                        "valid_mcq_accuracy": num_correct_val / len(valid_dataloader_mcq)
                    })
                else:
                    wandb.log({
                        "epoch": epoch,
                        "valid_mcq_loss": total_loss_valid_mcq / len(valid_dataloader_mcq),
                        "valid_mcq_accuracy": num_correct_val / len(valid_dataloader_mcq)
                    })

        
        logging.info("Evaluating the LM task")
        if not using_deepspeed or (using_deepspeed and local_rank == 0):
            total_loss_valid_lm = validate_lm(model, valid_dataloader_lm, using_deepspeed=using_deepspeed)

            logging.info(f"Avg valid loss: {total_loss_valid_lm / len(valid_dataloader_lm)}")

            if using_wandb:
                if lm_task_p != 0:
                    wandb.log({
                        "epoch": epoch,
                        "train_lm_loss": total_epoch_loss_lm / total_lm_questions,
                        "valid_lm_loss": total_loss_valid_lm / len(valid_dataloader_lm)
                    })
                else:
                    wandb.log({
                        "epoch": epoch,
                        "valid_lm_loss": total_loss_valid_lm / len(valid_dataloader_lm)
                    })

        model.train()


        if lm_task_p == 1:
            total_epoch_loss = total_epoch_loss_lm
            denom = len(valid_dataloader_lm)
        elif lm_task_p == 0:
            total_epoch_loss = total_epoch_loss_mcq
            denom = len(valid_dataloader_mcq)
        else:
            total_epoch_loss = total_epoch_loss_lm + total_epoch_loss_mcq
            denom = len(valid_dataloader_lm) + len(valid_dataloader_mcq)

        if (total_epoch_loss / denom) < best_val_loss:
            logging.info(f"EPOCH {epoch}: New best model with loss {total_epoch_loss / denom}")
            best_val_loss = total_epoch_loss / denom
            if early_stopping > 0:
                patience = early_stopping

            epoch_directory = os.path.join(output_dir, f'epoch_{epoch + 1}')
            os.makedirs(epoch_directory, exist_ok=True)
            # also save the model
            
            logging.info(f"Saving model at {epoch_directory}")
            #model.to("cpu")
            if using_deepspeed and deepspeed.ops.adam.cpu_adam: # using ZeRO
                model.save_checkpoint(epoch_directory, save_latest=True)
                if convert_to_fp32_now: # uses a lot of memory
                    state_dict = get_fp32_state_dict_from_zero_checkpoint(epoch_directory)
                    torch.save(state_dict, os.path.join(epoch_directory, 'model.pt'))
            else:
                model.save_checkpoint(epoch_directory)

        else:
            logging.info(f"No improvement, the best model still has loss {best_val_loss}")
            if early_stopping > 0:
                patience -= 1
                if patience == 0:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break

    if not using_deepspeed or (using_deepspeed and args.local_rank == 0):
        logging.info("Evaluating on test set")
        total_loss_test_lm = validate_lm(model, test_dataloader_lm, using_deepspeed=using_deepspeed, local_rank=f"cuda:{args.local_rank}")
        logging.info(f"FINAL TEST LOSS: {total_loss_test_lm / len(test_dataloader_lm)}")
        with open(f"./{fold}_fold_result.txt", "w") as f:
            f.write(f"FINAL TEST LOSS: {total_loss_test_lm / len(test_dataloader_lm)}")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a model on commonsense datasets of different types")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name, or path to a saved model")
    parser.add_argument("--dataset_type", type=str, default="metaphors", choices=["metaphors", "emotions", "physical"], help="Type of dataset")
    parser.add_argument("--specific_dataset", type=str, help="Specific dataset to use (will not train on all datasets of a type)")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--training_style", default="all_options", choices=["correct_options", "all_options"], help="Training style to adapt MC to LMHead model")
    parser.add_argument("--early_stopping", type=int, default=0, help="Use early stopping with patience of this many epochs")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train for. Mutually exclusive with early stopping.")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--debug", action="store_true", help="Train on a small number of examples")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 training")
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser.add_argument("--deepspeed", action="store_true", help="Use deepspeed for training")
    parser.add_argument("--is_local_model", action="store_true", help="Start training from a locally saved model. You need to specify this for training to work.")
    parser.add_argument("--lm_percentage", default=0, type=float, help="Will multitask between commonsense reasoning (MCQ) training and LM training, where p represents LM percentage of batches. Please check lr in the deepspeed config if doing continued lm training on an already trained model.")
    parser.add_argument("--book4_cv", action="store_true", help="Run cross validation for perplexity on the 4th book of harry potter.")
    parser.add_argument("--log_steps_freq", type=int, help="Number of steps after which to log (plots will be made for both epochs and steps)", default=1000)
    parser.add_argument("--wandb_offline", action="store_true", help="Run wandb offline")
    args, unknown = parser.parse_known_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    torch.cuda.set_device(args.local_rank)
    if args.local_rank != -1:
        deepspeed.init_distributed()

    if args.wandb and (args.local_rank == -1 or args.local_rank == 0):
        wandb.login()
        if args.specific_dataset is not None:
            if args.wandb_offline:
                run = wandb.init(mode="offline", project="commonsense_lm_training", config=args, name=f"{args.model_name}_{args.specific_dataset}_{args.training_style}")
            else:
                run = wandb.init(project="commonsense_lm_training", config=args, name=f"{args.model_name}_{args.specific_dataset}_{args.training_style}")
        else:
            if args.wandb_offline:
                run = wandb.init(mode="offline", project="commonsense_lm_training", config=args, name=f"{args.model_name}_{args.dataset_type}_{args.training_style}")
            else:
                run = wandb.init(project="commonsense_lm_training", config=args, name=f"{args.model_name}_{args.dataset_type}_{args.training_style}")

    if not args.output_dir:
        if args.specific_dataset is not None:
            args.output_dir = f"finetuned_models/{args.model_name}_{args.specific_dataset}_{args.training_style}"
        else:
            args.output_dir = f"finetuned_models/{args.model_name}_{args.dataset_type}_{args.training_style}"
        
        logging.info(f"Saving to {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)

    commonsense_datasets = {
        "metaphors": ["nightingal3/figqa", "ColumbiaNLP/FLUTE"],
        "emotions": ["social_i_qa"],
        "physical": ["piqa", "hellaswag"]
    }

    text_dataset = {
        #"debug": "roneneldan/TinyStories",
        "c4": "c4"
    }

    logging.info("Loading datasets")
    if args.specific_dataset is not None:
        sel_datasets = [args.specific_dataset]
    else:
        sel_datasets = commonsense_datasets[args.dataset_type]

    if args.lm_percentage > 0:
        assert args.lm_percentage <= 1, "lm percentage must be in [0, 1]"

    if args.lm_percentage == 1:
        logging.info("lm_percentage of 1 passed, this run will just run language modelling training.")
    elif args.lm_percentage == 0:
        logging.info("lm_percentage of 0 passed, this will just finetune on the commonsense task. The model will not be usable as an LM afterwards.")
    
    hf_datasets_train = [datasets.load_dataset(dataset_name, split="train") for dataset_name in sel_datasets]
    hf_dataset_valid = [datasets.load_dataset(dataset_name, split="validation") for dataset_name in sel_datasets]
    #text_dataset_train = datasets.load_dataset(text_dataset["debug"], split="train")
    #text_dataset_valid = datasets.load_dataset(text_dataset["debug"], split="validation")

    if args.debug:
        hf_datasets_train = [ds.select(range(100)) for ds in hf_datasets_train]
        hf_dataset_valid = [ds.select(range(100)) for ds in hf_dataset_valid]

    dschf = None
    if args.deepspeed:
        dschf = "ds_config.json"

    logging.info("Loading model")
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    if model.num_parameters() > 1e9:
        # init deepspeed
        logging.info("Large model, will use deepspeed for training")
        #ds_conf = HfDeepSpeedConfig("ds_config.json")
       #engine = deepspeed.initialize(model=model, config_params="ds_config.json")
        using_deepspeed = True
    else:
        model.to(DEVICE)
        using_deepspeed = args.deepspeed

    if not args.is_local_model:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    # only needed for GPT-2
    tokenizer.pad_token = tokenizer.eos_token

    if args.book4_cv:
        logging.info("Performing cross validation on book 4")
        book4 = process_book_file("./harry_potter/book1_fixed.txt", tokenizer)
        kfold = KFold(n_splits=3, shuffle=True, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(kfold.split(book4)):
            if fold != 2:
                continue
            logging.info(f"=== Fold {fold} ===")
            book4_train = [book4[i] for i in train_idx]
            book4_test = [book4[i] for i in test_idx]
            book4_test, book4_valid = train_test_split(book4_test, test_size=0.5)

            if args.debug: 
                book4_train, book4_test, book4_valid = book4_train[:100], book4_test[:100], book4_valid[:100]

            book_train_dataset = datasets.Dataset.from_dict({"text": book4_train})
            book_valid_dataset = datasets.Dataset.from_dict({"text": book4_valid})
            book_test_dataset = datasets.Dataset.from_dict({"text": book4_test})

            trained_model = train_all_options(model, tokenizer, hf_datasets_train, hf_dataset_valid, sel_datasets, book_train_dataset, book_valid_dataset, book_test_dataset, early_stopping=args.early_stopping, lm_task_p=1, fp16=args.fp16, dsconfig=dschf, using_deepspeed=using_deepspeed, using_wandb=args.wandb, local_rank=args.local_rank, output_dir=f"{args.output_dir}/fold_{fold}", n_epochs=args.num_epochs, log_steps_frequency=args.log_steps_freq, fold=fold)

    else:
        if args.training_style == "correct_options":
            model = train_correct_options_only(model, tokenizer, hf_datasets_train, hf_dataset_valid, early_stopping=args.early_stopping, lm_task_p=args.lm_percentage, fp16=args.fp16, dsconfig=dschf, using_deepspeed=using_deepspeed, using_wandb=args.wandb, output_dir=args.output_dir)
        elif args.training_style == "all_options":
            model = train_all_options(model, tokenizer, hf_datasets_train, hf_dataset_valid, sel_datasets, text_dataset_train, text_dataset_valid, early_stopping=args.early_stopping, lm_task_p=args.lm_percentage, fp16=args.fp16, dsconfig=dschf, using_deepspeed=using_deepspeed, using_wandb=args.wandb, local_rank=args.local_rank, output_dir=args.output_dir, n_epochs=args.num_epochs, log_steps_frequency=args.log_steps_freq)

    if args.wandb and (args.local_rank == -1 or args.local_rank == 0):
        run.finish()


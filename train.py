import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.optim import AdamW
from datasets import load_dataset  
from transformers import get_linear_schedule_with_warmup


model_name = "" 
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)


train_dataset_path = ""
eval_dataset_path = ""

train_dataset = load_dataset("json", data_files=train_dataset_path)["train"]
eval_dataset = load_dataset("json", data_files=eval_dataset_path)["train"]


def preprocess_function(examples):
    prefix = "Format your answer as closely as possible to the provided calculation example and solve the following math problem step-by-step"
    

    reasoning_inputs = [prefix + " " + question for question in examples["question"]]
    answer_targets = examples.get("answer", [""] * len(examples["question"]))  
    

    reasoning_inputs = [str(a) for a in reasoning_inputs]
    answer_targets = [str(a) for a in answer_targets]


    reasoning_encodings = tokenizer(reasoning_inputs, max_length=512, truncation=True, padding="max_length")
    answer_encodings = tokenizer(answer_targets, max_length=256, truncation=True, padding="max_length")
    
    return {
        "input_ids": reasoning_encodings["input_ids"],
        "attention_mask": reasoning_encodings["attention_mask"],
        "labels": answer_encodings["input_ids"]  
    }


tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)


training_args = TrainingArguments(
    output_dir="",
    eval_strategy="steps",  
    eval_steps=100,
    save_steps=100,
    save_total_limit=3,
    learning_rate=1e-5,  
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    weight_decay=0.01,
    fp16=False,
    push_to_hub=False,
    load_best_model_at_end=True,
    remove_unused_columns=False, 
    lr_scheduler_type="constant"  
)


optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)


num_training_steps = len(tokenized_train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)


class CustomLossTrainer(Trainer):
    def __init__(self, *args, optimizer=None, scheduler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        return self.optimizer, self.scheduler

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.accelerator.accumulate(model):
            loss = self.compute_loss(model, inputs)

            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.deepspeed:
                self.deepspeed.backward(loss)
            else:
                loss.backward()

        return loss.item()  

    def compute_loss(self, model, inputs):
        reasoning_outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=inputs["labels"])
        reasoning_loss = reasoning_outputs.loss

       
        answer_loss = reasoning_loss 

        alpha = 0.8
        weighted_loss = alpha * reasoning_loss + (1 - alpha) * answer_loss

        return weighted_loss


trainer = CustomLossTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    optimizers=(optimizer, scheduler), 
)

trainer.train()

results = trainer.evaluate(eval_dataset=tokenized_eval_dataset)
print("Test Set Evaluation Results:", results)

# load tokenizer
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AdamW

context_length = 512
tokenizer = AutoTokenizer.from_pretrained("./new_tokenizer")


# load from disk
tokanized_train_dataset = load_from_disk("tokanized_train_dataset")
tokanized_test_dataset = load_from_disk("tokanized_test_dataset")

# shuffle the datasets
# tokanized_train_dataset = tokanized_train_dataset.shuffle()
# tokanized_test_dataset = tokanized_test_dataset.shuffle()


scale = 1

# Initializing a GPT2 configuration
configuration = GPT2Config(
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    n_embd=int(768 * scale),
    n_layer=int(12 * scale),
    n_head=int(12 * scale),
)

# Initializing a model from the configuration
model = GPT2LMHeadModel(configuration)

# Accessing the model configuration
configuration = model.config

model_size = sum(t.numel() for t in model.parameters())
print(f"ICE GPT-2 size: {model_size/1000**2:.1f}M parameters")


tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, pad_to_multiple_of=None
)


optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)

batch_size = 4

args = TrainingArguments(
    output_dir="icebreaker",
    overwrite_output_dir=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="steps",
    # eval_steps=5_000,
    eval_steps=10,
    logging_steps=20,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    # warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
)

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokanized_train_dataset,
    eval_dataset=tokanized_test_dataset,
    data_collator=data_collator,
    optimizers=(optimizer, None),
)

trainer.train(resume_from_checkpoint=True)

model.save_pretrained("icebreaker")
tokenizer.save_pretrained("icebreaker")

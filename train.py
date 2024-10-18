import torch
import torch.nn as nn
from torch.utils.data import DataLoader , dataset , random_split
from torch.optim.lr_scheduler import LambdaLR

## importing the different module 
from Dataset import LingualDataSet , causal_mask
from Model import Develop_transformer
from config import get_config , save_weights_file_path

import os
import warnings



from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

## importing the tensorboard 
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

from tqdm import tqdm



def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

# code taken from: https://huggingface.co/docs/tokenizers/quicktour

def get_or_build_tokenizer(config, ds, lang):
    ## config[tokenizers_file] = "---/tokenizers/tokenizer_{0}.jason"
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer 


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    #decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    decoder_input = torch.full((1, 1), sos_idx, dtype=torch.long).to(device)
    while True:
        if decoder_input.size(1) >= max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        #decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
        
        # Update decoder input for the next token
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0).unsqueeze(0).to(device)], dim=1)


        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)



def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0
    source_texts = []
    expected = []
    predicted = []
    console_width = 80

    with torch.inference_mode():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output on consol
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()        






def take_DS(config):
    ## dataset from https://huggingface.co/datasets/Helsinki-NLP/opus_books  
    ## English to Spenish
    ds_raw =  load_dataset("Helsinki-NLP/opus_books",f"{config['srs_lang']}-{config['tgt_lang']}",split='train')

    ## Develop the Tokenizers
    Tokenizer_srs = get_or_build_tokenizer(config , ds_raw , config['srs_lang'])
    Tokenizer_tgt = get_or_build_tokenizer(config , ds_raw , config['tgt_lang'])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = int(0.1 * len(ds_raw))

    train_ds_raw , val_ds_raw = random_split(ds_raw , [train_ds_size,val_ds_size])

    
    train_ds = LingualDataSet(train_ds_raw,Tokenizer_srs,Tokenizer_tgt,config["srs_lang"],config["tgt_lang"] , config["sql_len"])
    val_ds   = LingualDataSet(val_ds_raw,Tokenizer_srs,Tokenizer_tgt,config["srs_lang"],config["tgt_lang"] , config["sql_len"]) 

    # Debugging print
    print("Training dataset size:", len(train_ds_raw))
    print("Validation dataset size:", len(val_ds_raw))
 
    ## calculating the max len of src and tgt 
    max_len_srs = 0
    max_len_tgt = 0

    for obj in ds_raw:
        srs_len = Tokenizer_srs.encode(obj['translation'][config['srs_lang']]).ids
        tgt_len = Tokenizer_tgt.encode(obj['translation'][config['tgt_lang']]).ids
        max_len_srs = max(max_len_srs, len(srs_len))
        max_len_tgt = max(max_len_tgt , len(tgt_len))
    print(f"max length of   srs language sentence  is {max_len_srs} ")
    print(f"Max len of  tgt language sentence is {max_len_tgt}")    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds , batch_size=1, shuffle=True)

    return train_dataloader , val_dataloader , Tokenizer_srs , Tokenizer_tgt

def call_model(config , vocab_srs_len , vocab_tgt_len):
    model = Develop_transformer(vocab_srs_len , vocab_tgt_len , config['sql_len'] , config['sql_len'] )
    return model



device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Current using device {device}")
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")

    Path(config['model_folder']).mkdir(parents=True,exist_ok=True)
    train_dataloader , val_dataloader , Tokenizer_srs , Tokenizer_tgt = take_DS(config)   


    model = call_model(config , Tokenizer_srs.get_vocab_size(),Tokenizer_tgt.get_vocab_size()).to(device) 


    ## tensorboard 
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr =config['lr'],eps=1e-9)

    inti_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = save_weights_file_path(config , config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        #model.load_state_dict(state['model_state_dict'])
        inti_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=Tokenizer_srs.token_to_id('[PAD]'), label_smoothing=0.1).to(device)


    for epoch in range(inti_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")         

        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Print shapes for debugging
            '''
            print(f"Encoder input shape: {encoder_input.shape}")  # (B, seq_len)
            print(f"Decoder input shape: {decoder_input.shape}")  # (B, seq_len)
            print(f"Encoder mask shape: {encoder_mask.shape}")  # (B, 1, 1, seq_len)
            print(f"Decoder mask shape: {decoder_mask.shape}")  # (B, 1, seq_len, seq_len)

            '''

            
            # Run the tensors through the encoder, decoder, and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
           # print(f"Encoder output shape: {encoder_output.shape}")  # (B, seq_len, d_model)

            decoder_output = model.decode(encoder_output, decoder_input, encoder_mask, decoder_mask)  # (B, seq_len, d_model)
            #print(f"Decoder output shape: {decoder_output.shape}")  # (B, seq_len, d_model)

            proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)
           # print(f"Projected output shape: {proj_output.shape}")  # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device)  # (B, seq_len)
           # print(f"Label shape: {label.shape}")  # (B, seq_len)
            
            '''
             # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            '''
            

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, Tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()
            optimizer.zero_grad(set_to_none = True)
                         
            # Update the weights
            optimizer.step()
            

            

            # Run validation at the end of every epoch
           # run_validation(model, val_dataloader,  Tokenizer_srs ,  Tokenizer_tgt, config['sql_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)
            global_step += 1
            ## save the model at every epoch 
            # Save the model at the end of every epoch
        model_filename = save_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)        















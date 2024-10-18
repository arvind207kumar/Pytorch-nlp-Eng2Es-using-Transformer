import torch
import torch.nn as nn
from torch.utils.data import DataLoader , Dataset


### user define tokenizer for  unserstanding 
'''
import torch
from torch.utils.data import DataLoader
import random

# Mock Tokenizer class
class MockTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)
    
    def encode(self, text):
        # Tokenizes text into word IDs. For simplicity, we will convert each character to a token ID.
        return TokenizedResult([self.vocab.get(char, self.vocab['[UNK]']) for char in text])

    def token_to_id(self, token):
        # Returns the token ID
        return self.vocab.get(token, self.vocab['[UNK]'])

# Mock TokenizedResult class
class TokenizedResult:
    def __init__(self, ids):
        self.ids = ids

# Causal mask function (used for decoder)
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

# Sample dataset
sample_data = [
    {
        'translation': {
            'en': 'hello',
            'fr': 'bonjour'
        }
    },
    {
        'translation': {
            'en': 'world',
            'fr': 'monde'
        }
    },
    {
        'translation': {
            'en': 'thank you',
            'fr': 'merci'
        }
    }
]

# Example vocabulary for English and French, plus special tokens
vocab_src = {'h': 1, 'e': 2, 'l': 3, 'o': 4, 'w': 5, 'r': 6, 'd': 7, 't': 8, 'a': 9, 'n': 10, 'k': 11, 'y': 12, 'u': 13, '[SOS]': 14, '[EOS]': 15, '[PAD]': 0, '[UNK]': 16}
vocab_tgt = {'b': 1, 'o': 2, 'n': 3, 'j': 4, 'u': 5, 'r': 6, 'm': 7, 'e': 8, 'c': 9, 'i': 10, '[SOS]': 14, '[EOS]': 15, '[PAD]': 0, '[UNK]': 16}

# Tokenizers
tokenizer_srs = MockTokenizer(vocab_src)
tokenizer_tgt = MockTokenizer(vocab_tgt)



class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_srs, tokenizer_tgt, srs_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_srs = tokenizer_srs
        self.tokenizer_tgt = tokenizer_tgt
        self.srs_lang = srs_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.srs_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_srs.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Create encoder input with <s>, tokens, </s>, and padding
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Create decoder input with <s>, tokens, and padding
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Create label with tokens, </s>, and padding
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

# Create an instance of the dataset
seq_len = 10
dataset = BilingualDataset(sample_data, tokenizer_srs, tokenizer_tgt, 'en', 'fr', seq_len)

# Create DataLoader to batch data
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Visualize one batch of data
for batch in dataloader:
    print("Source Texts:", batch["src_text"])
    print("Target Texts:", batch["tgt_text"])
    print("Encoder Input:\n", batch["encoder_input"])
    print("Decoder Input:\n", batch["decoder_input"])
    print("Labels:\n", batch["label"])
    print("Encoder Mask:\n", batch["encoder_mask"])
    print("Decoder Mask:\n", batch["decoder_mask"])
    break  # Visualize only one batch for simplicity


'''





'''
def causal_mask(size):
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
        return mask == 0

class LingualDataSet(Dataset):
    def __init__(self , ds_raw , tokenizer_srs , tokenizer_tgt , srs_lang , tgt_lang , sql_len):
        super().__init__()
        self.ds_raw = ds_raw
        self.tokenizer_srs = tokenizer_srs
        self.tokenizer_tgt = tokenizer_tgt
        self.srs_lang = srs_lang
        self.tgt_lang = tgt_lang
        self.sql_len = sql_len
        
        ## geting the token id of sos , eos , pad by using the tokenizer_tgt instances and thand callinf the token_to_id class 
        # to get the id in the tensor form of dtype torch.int64
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(ds_raw)

    def __getitem__(self, index) :

        src_tgt_pair  = self.ds_raw[index]

        src_txt = src_tgt_pair['translation'][self.srs_lang]
        tgt_txt = src_tgt_pair['translation'][self.tgt_lang]

        ## Transform the txt to token   
        enc_input_tokens = self.tokenizer_srs.encode(src_txt).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_txt).ids


        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # add <s> and </s> so reduce by 2 
        
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 ## add <s> only so rduce by 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:  ## conditio to check the sql_len is greater than enc_token

            raise ValueError("Sentence is too long")
        
        # Add <s> and </s> token to the encoder toke 

        encoder_input = torch.cat(
            [self.sos_token,
             torch.Tensor(enc_input_tokens,dtype = torch.int64),
             self.eos_token,
             torch.Tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),

            ],
            dim = 0,
        )
        

        ## Add <s> only token to the decoder input 
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.Tensor(dec_input_tokens , dtype = torch.int64),
                torch.Tensor([self.pad_token] * dec_num_padding_tokens , dtype = torch.int64),

            ],
            dim = 0,
        )

        ## ADD the </s> to the end of token and reduce the size by 1
        label_input =torch.cat(
            [
                torch.Tensor(dec_input_tokens , dtype = torch.int64),
                self.eos_token,
                torch.Tensor([self.pad_token] * dec_num_padding_tokens , dtype = torch.int64),


            ], dim =0,
        )

        assert encoder_input.size(0) == self.seq_len, " encoder_input len is matching with sql_len "
        assert decoder_input.size(0) == self.seq_len , " decoder_input len is matching with sql_len "
        assert label_input.size(0) == self.seq_len , " label_input len is matching with sql_len "

       
        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label_input,  # (seq_len)
            "src_text": src_txt,
            "tgt_text": tgt_txt,
        } 

    



'''
        
import torch
from torch.utils.data import Dataset

def causal_mask(size):
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
        return mask == 0

class LingualDataSet(Dataset):
    def __init__(self, ds_raw, tokenizer_srs, tokenizer_tgt, srs_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds_raw = ds_raw
        self.tokenizer_srs = tokenizer_srs
        self.tokenizer_tgt = tokenizer_tgt
        self.srs_lang = srs_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        # Getting the token id of sos, eos, pad by using the tokenizer_tgt instances
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds_raw)
    
    def truncate_tokens(self, tokens, max_len):
        """ Truncate tokens to ensure they don't exceed the max length. """
        return tokens[:max_len] if len(tokens) > max_len else tokens

    def __getitem__(self, index):
        src_tgt_pair = self.ds_raw[index]
        src_txt = src_tgt_pair['translation'][self.srs_lang]
        tgt_txt = src_tgt_pair['translation'][self.tgt_lang]

        # Transform the text to tokens
        enc_input_tokens = self.tokenizer_srs.encode(src_txt).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_txt).ids


        '''

       
        
        '''

        # Truncate tokens if they're too long
        enc_input_tokens = self.truncate_tokens(enc_input_tokens, self.seq_len - 2)  # for <SOS> and <EOS>
        dec_input_tokens = self.truncate_tokens(dec_input_tokens, self.seq_len - 1)  # for <SOS>
        
        
        # Recalculate padding tokens after truncation
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # Add <SOS> and <EOS>
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # Add <SOS>

        '''       
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:  # condition to check the seq_len is greater than enc_token
            raise ValueError("Sentence is too long")


        '''
        
        # Add <s> and </s> token to the encoder tokens
        encoder_input = torch.cat(
            [self.sos_token,
             torch.tensor(enc_input_tokens, dtype=torch.int64),
             self.eos_token,
             torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)],
            dim=0,
        )

        # Add <s> only token to the decoder input
        decoder_input = torch.cat(
            [self.sos_token,
             torch.tensor(dec_input_tokens, dtype=torch.int64),
             torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)],
            dim=0,
        )

        # Add </s> to the end of tokens and reduce the size by 1
        label_input = torch.cat(
            [torch.tensor(dec_input_tokens, dtype=torch.int64),
             self.eos_token,
             torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)],
            dim=0,
        )

        assert encoder_input.size(0) == self.seq_len, "encoder_input length is not matching with seq_len"
        assert decoder_input.size(0) == self.seq_len, "decoder_input length is not matching with seq_len"
        assert label_input.size(0) == self.seq_len, "label_input length is not matching with seq_len"

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # (1, seq_len) & (1, seq_len, seq_len)
            "label": label_input,  # (seq_len)
            "src_text": src_txt,
            "tgt_text": tgt_txt,
        }

    
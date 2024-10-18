import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self,d_model:int,voca_size:int):
        super().__init__()
        self.d_model = d_model
        self.voca_size = voca_size
        self.embedding = nn.Embedding(voca_size,d_model)

    def forward(self  , x):
        return self.embedding(x)   * math.sqrt(self.d_model)
    


    
class PositionEncodding(nn.Module):
     def __init__(self,d_model:int,seqn_len:int,dropout:float)->None:

        super().__init__()
        self.d_model = d_model
        self.seqn_len = seqn_len
        self.Dropout = nn.Dropout(dropout)

        ## Creating the matrix of length sqqn_len and d_model
        pe = torch.zeros(seqn_len,d_model)

        ## creating the sequeens length matrix of shape (seqn_len , 1)
        # position = torch.arange(0,seqn_len,dtype = torch.float).unsqueeze(1) 
        # position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term) 

        for pos in range(seqn_len):
            for i in range(0,d_model,2):
                #pe[pos,i] = math.sin(pos / (10000 ** ((2*i)/d_model))) 
                #pe[pos , i+1]  = math.sin(pos / (10000 ** (2*(i+1)/d_model)))
                pe[pos, i] = math.cos(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        #pe.unsqueeze(0).transpose(0,1) 
        pe = pe.unsqueeze(0)

        ##saving the parameter in register buffer that not updated during the training
        
        self.register_buffer("pe", pe)       

     def forward(self , x):
         x = x+ (self.pe[:,:x.shape[1],:]).requires_grad_(False)
         return self.Dropout(x)





class LayerNormalization(nn.Module):
    def __init__(self,features:int, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        #self.beta = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(features))

        #self.register_parameter('alpha', nn.Parameter(torch.ones(1)))
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    



class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int , d_ff:int, dropout:float)->None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model)

    def forward(self , x):
        ## (batch_size , seq_len , d_model) --> (batch_size , seq_len , d_ff) --> (batch_size , seq_len , d_model)
        return  self.linear_2(self.dropout(nn.ReLU()(self.linear_1(x))))    



        

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, H:int, dropout_rate:float) -> None:
        super().__init__()
        self.s__model =d_model
        self.dropout = nn.Dropout(dropout_rate)
        self.H = H
        assert d_model % H ==0 , "d_model is not divisible by provided No of Head"

        self.d_head = d_model//H   ## dimention of each head 

        # Linear projections for query, key, and value   
        
        self.quary_linear= nn.Linear(d_model,d_model , bias = False)  
        self.key_linear= nn.Linear(d_model,d_model , bias = False)  
        self.value_linear = nn.Linear(d_model, d_model  , bias = False) 

        ##output Linear projection
        self.output_linear = nn.Linear(d_model , d_model , bias = False)

        ## Dropout layer 
        self.Dropout = nn.Dropout(dropout_rate)


    @staticmethod
    def attention(query , key , value , mask , dropout:nn.Dropout):

        """
        Computes the scaled dot-product attention.
        
        Args:
        - query: The query matrix.
        - key: The key matrix.
        - value: The value matrix.
        - mask: Mask for ignoring certain positions (e.g., padding tokens).
        - dropout: Dropout applied after softmax to attention scores.
        
        Returns:
        - The output after attention is applied.
        - The attention scores (for visualization).
        """

        d_head = query.shape[-1]

        # Compute the scaled dot-product attention
       
        ## shape of query key  is (batch_size ,H , seq_len , d_head ) and  key shape is (batch_size ,H , seq_len , d_head )  
        # key is transpose to (batch_size ,H , seq_len , d_head )  --> (batch_size , H , d_head , sql_len)
        attention_scores = (query @ key.transpose(-1,-2)) / math.sqrt(d_head)
        ## shape after multiplication (batch_size , H , sql_len(its a sql_len of query part) , sql_len(sequence length of key part ))

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        ## calculating the sofmax probability of attention_score accros last dimentio -1(allong the seq_len of key )
        attention_scores = attention_scores.softmax(dim = -1)  


        if dropout is not None:
            attention_scores = dropout(attention_scores) 

        ## shape of attention_score is (batch_size , H , sql_len(its a sql_len of query part) , sql_len(sequence length of key part ))
        ## shape of value (batch_size , H , seq_len , d_head)
        ### (attention_scores  @ value) shape become (batch_size , H ,seq_len , d_head )
        return (attention_scores  @ value) , attention_scores   





    def forward(self ,q , k , v, mask ):
        
        query =  self.quary_linear(q)  ### shape of query is (batch_size , seq_len , dimention_of_embaded=d_model(512))
        key  = self.key_linear(k)      ### shape of key is (batch_size , seq_len , dimention_of_embaded=d_model(512))

        value = self.value_linear(v)   ### shape of value is (batch_size , seq_len , dimention_of_embaded=d_model(512))
       # print("Query initial shape:", query.shape)  # Batch size, seq_len, d_model


        query = query.view(query.shape[0], query.shape[1], self.H , self.d_head).transpose(1,2)  
        key =key.view(key.shape[0], key.shape[1], self.H, self.d_head).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.H , self.d_head).transpose(1,2)

       # print("Query shape after reshaping:", query.shape)  # Should be [batch_size, H, seq_len, d_head]

        ## shape convertion of query , key and value  by view  and acomodetingthe two new dimention which is d_H and d_model(embedded dimention)
        #  (batch_size , seq_len , d_model) --> 
        # (batch_size , seq_len , H(no. of Head in multiHeadAttention), d_head(dimention of each Head of multiHeadAttention getting by dividing D_model//H))
        ## aftter transpose the shape (batch_size, seq_len , H, d_head) --> (batch_size , H , seq_len , d_head)
       
        ## shape of query , key and value after  above operation is (batch_size , H , seq_len , d_head)

        # Step 3: Calculate attention for each head
        # Query, key, value have shape: (batch, H, seq_len, d_k)
        # The attention method will return the attended values and the attention scorestionBlock.attention(query, key, value, mask, self.dropout)

        X , self.attention_score = MultiHeadAttentionBlock.attention(query, key , value , mask , self.dropout)
        ## X shape =  (batch_size , H ,seq_len , d_head )
        ## Attention_score shape =  (batch_size , H , sql_len , sql_len)

        # Step 4: Reshape the outputs back to the original shape
        # Transpose: (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k)
        # Contiguous and reshape: (batch, seq_len, h * d_k) -> (batch, seq_len, d_model)

        ## (batch_size , sql_len, H , d_head) --> (batch_size , sql_len , self.H(8) * self.d_head(64))  
        # which is converted into shape of (batch_size  , sql_len , d_model)   H*d_head = d_model (8*64==512(d_model))
        #    
        X = X.transpose(1,2).contiguous().view(X.shape[0],-1 , self.H * self.d_head)



        # Step 5: Final linear projection (output_linear) to get the output back to the original d_model
        # Shape remains (batch, seq_len, d_model)
        return  self.output_linear(X)
    



class ResidualConnection(nn.Module): 
    def __init__(self,features:int, dropout_rate:float)->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = LayerNormalization(features)

    def forward(self , x, sublayer):
        return  x + self.dropout(sublayer(self.norm(x)))
    




class EncoderBlock(nn.Module):
    def __init__(self ,feature:int, attention_block :MultiHeadAttentionBlock , feed_forward_block : FeedForwardBlock , dropout : float )->None:
        super().__init__()
        
        self.attention_block = attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_block = nn.ModuleList(ResidualConnection(feature,dropout) for _ in range(2))

    def forward(self , X , srs_mask):  
        ## srs_mask this mask we want to  apply to input of encoder  
        #we  need this b/c we want to hide the interection of padding word with other word 

        X = self.residual_block[0](X , lambda X : self.attention_block(X, X , X , srs_mask))

        X = self.residual_block[1](X , lambda X : self.feed_forward_block(X))

        return X
    

class Encoder(nn.Module):

    def __init__(self,features:int , layers:nn.ModuleList)->None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self , X , mask):
        for layer in self.layers:
            X = layer(X , mask)
        return self.norm(X)    




## Decoder Block 
class DecoderBlock(nn.Module):
    def __init__(self,features:int, self_attention_block : MultiHeadAttentionBlock , cross_attention_block:MultiHeadAttentionBlock , feed_forward_block : FeedForwardBlock,dropout_rate:float)->None:
        super().__init__()
        self.self_attention_block  =  self_attention_block
        self.cross_attention_block =  cross_attention_block
        self.feed_forward_block    =  feed_forward_block

        self.dropout=nn.Dropout(dropout_rate)

        self.residual_connection = nn.ModuleList(ResidualConnection(features=features,dropout_rate=dropout_rate) for _ in range(3))


    def forward(self ,encoder_output, X , srs_mask, targ_mask ):

        X = self.residual_connection[0](X , lambda X : self.self_attention_block(X,X,X,targ_mask))
        X = self.residual_connection[1](X, lambda X : self.cross_attention_block(X , encoder_output, encoder_output, srs_mask))
        X = self.residual_connection[2](X , lambda X : self.feed_forward_block(X))

        return X


## Decoder STACK  OF LAYER

class Decoder(nn.Module):
    def __init__(self,features:int,layers : nn.ModuleList)->None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    def  forward(self ,encoder_output, X , srs_mask, targ_mask):
        for layer in self.layers:
            X = layer(encoder_output, X , srs_mask, targ_mask)
        return self.norm(X)    


class ProjecLayer(nn.Module):
    def __init__(self,d_model:int , vocb_size:int)->None:
        super().__init__()
        self.proj = nn.Linear(d_model , vocb_size)

    def forward(self , X):
        ## (batch_size , seq_len , d_model)    -->> (batch_size , seq_len , vocb_size) projection on the vocab_size 
        return torch.log_softmax(self.proj(X),dim=-1)
    

class Transformer(nn.Module):
    def __init__(self , encoder: Encoder , decoder : Decoder , srs_embed:InputEmbedding, tgt_embed: InputEmbedding , srs_pos : PositionEncodding , tgt_pos : PositionEncodding, projection_layer : ProjecLayer)->None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.srs_embed = srs_embed
        self.tgt_embed = tgt_embed
        self.srs_pos = srs_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer   

    def encode(self , srs , srs_mask):
        # (batch, seq_len, d_model)
        srs = self.srs_embed(srs)
        srs = self.srs_pos(srs)
        return self.encoder(srs,srs_mask)
    
    def decode(self , encoder_output : torch.Tensor , tgt:torch.Tensor, srs_mask:torch.Tensor , tgt_mask:torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(encoder_output, tgt , srs_mask, tgt_mask)
    
    def project(self , X ):
         # (batch, seq_len, vocab_size)
        return self.projection_layer(X)
    


        
def Develop_transformer(srs_vocab_size: int, tgt_vocab_size: int, srs_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    ## considering all the HyperParameter that being used in Transformer

    ## Creating the input Embedding layer
    srs_embed = InputEmbedding(d_model, srs_vocab_size)
    tgt_embed = InputEmbedding(d_model,tgt_vocab_size)


    ## Creating the Positional Embedding Layer
    srs_position_embed  = PositionEncodding(d_model,srs_seq_len,dropout)
    tgt_position_embed = PositionEncodding(d_model, tgt_seq_len , dropout) 

    def forward(self, encoder_input, encoder_mask, decoder_input, decoder_mask):
        # Check the size before any significant reshaping
        print("Encoder input shape:", encoder_input.shape)
        print("Decoder input shape:", decoder_input.shape)

        # Proceed with the operation, checking input tensor shapes at each step
        # Reshape logic should explicitly account for the number of elements:
        expected_size = 1 * 1 * self.H * self.d_head  # Example based on how you set this up
        if encoder_input.nelement() != expected_size:
            print(f"Unexpected number of elements: {encoder_input.shape}")

    ## Creating the Encoder 
    encoder_blocks = []
    for _ in range(6):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h , dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff , dropout)
        encoder_block = EncoderBlock(d_model ,encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)


    ## Creating The Decoder 
    decoder_blocks = []
    for _ in range(6):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h , dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h , dropout)
        feed_forward_block = FeedForwardBlock(d_model , d_ff , dropout)
        decoder_block  = DecoderBlock(d_model, decoder_self_attention_block ,decoder_cross_attention_block,feed_forward_block, dropout  )   
        decoder_blocks.append(decoder_block)

    ## Creating the Encoder and Decoder 
    encoder = Encoder(d_model ,nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model , nn.ModuleList(decoder_blocks))

    ## Create the projection Layer
    projection_layer = ProjecLayer(d_model,tgt_vocab_size)

    ## create the transformer 
    transformer = Transformer(encoder , decoder , srs_embed , tgt_embed, srs_position_embed , tgt_position_embed , projection_layer  )   

    ## intialise the Parameters 
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer        


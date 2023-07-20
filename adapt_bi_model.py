import torch
import torch.nn as nn


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hidden_dim=768, pf_dim=768 * 2, dropout_ratio=0.1):
        super().__init__()

        self.fc_1 = nn.Linear(hidden_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]

        x = self.dropout(nn.GELU(self.fc_1(x)))
        # x: [batch_size, seq_len, pf_dim]

        x = self.fc_2(x)
        # x: [batch_size, seq_len, hidden_dim]

        return x


class Encode(nn.Module):
    def __init__(self):
        super(Encode, self).__init__()
        self.enc_self_attn = MultiHeadAttentionLayer()
        self.pos_ffn = PositionwiseFeedforwardLayer()

    def forward(self, enc_inputs, tgt_inputs, enc_self_attn_mask=None, CROSS=False):
        if CROSS:
            enc_outputs, attn = self.enc_self_attn(enc_inputs, tgt_inputs, tgt_inputs, enc_self_attn_mask)
        else:
            enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)

        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim=768, n_heads=8, dropout_ratio=0.1, device="cuda:0"):
        super().__init__()

        assert hidden_dim % n_heads == 0

        self.hidden_dim = hidden_dim  # 임베딩 차원
        self.n_heads = n_heads  # 헤드의 개수 : 서로 다른 어텐션 컨셉의 수
        self.head_dim = hidden_dim // n_heads  # 각 헤드에서의 임베딩 차원

        self.fc_q = nn.Linear(hidden_dim, hidden_dim)  # Query 값에 적용된 fc layer
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)  # Key 값에 적용될 fc layer
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)  # Value 값에 적용될 fc layer

        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query: [batch_size, query_len, hidden_dim]
        # key: [batch_size, key_len, hidden_dim]
        # value: [batch_size, value_len, hidden_dim]


        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q: [batch_size, query_len, hidden_dim]
        # K: [batch_size, key_len, hidden_dim]
        # V: [batch_size, value_len, hidden_dim]

        # hidden_dim -> n_heads X head_dim 형태로 변형
        # n_heads(h)개의 서로 다른 어텐션 컨셉을 학습

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q: [batch_size, n_heads, query_len, head_dim]
        # K: [batch_size, n_heads, key_len, head_dim]
        # V: [batch_size, n_heads, value_len, head_dim]

        # Attention 계산
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy: [batch_size, n_heads, query_len, key_len]
        # 마스크를 사용하는 경우
        if mask is not None:
            # 마스크 값이 0인 부분을 -1e10으로 채우기
            energy = energy.masked_fill(mask == 0, -1e10)

        # 어텐션 스코어 계산: 각 단어에 대한 확률 값
        attention = torch.softmax(energy, dim=-1)

        # attention: [batch_size, n_heads, query_len, key_len]

        # Scaled Dot-Product Attention 계산

        x = torch.matmul(self.dropout(attention), V)
        # x: [batch_size, n_heads, query_len, head_dim]

        x = x.permute(0, 2, 1, 3).contiguous()
        # x: [batch_size, query_len, n_heads, head_dim]

        x = x.view(batch_size, -1, self.hidden_dim)
        # x: [batch_size, query_len, hidden_dim]

        x = self.fc_o(x)
        # x: [batch_size, query_len, hidden_dim]

        return x, attention


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size=768, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertSelfOutput(nn.Module):
    def __init__(self, hidden_size=768, hidden_dropout_prob=0.1):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self):
        super(BertAttention, self).__init__()
        self.mt = MultiHeadAttentionLayer()
        self.output = BertSelfOutput()

    def forward(self, input_tensor, audio_data, attention_mask = None):
        self_output, _ = self.mt(input_tensor, audio_data, audio_data, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, intermediate_size=3072, hidden_size=768, hidden_dropout_prob=0.1):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self):
        super(BertLayer, self).__init__()
        self.attention = BertAttention()
        self.intermediate = BertIntermediate()
        self.output = BertOutput()

    def forward(self, hidden_states, all_audio_data, attention_mask=None):
        attention_output = self.attention(hidden_states, all_audio_data, attention_mask)

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, num_attention_heads=8):
        super(BertEncoder, self).__init__()
        layer = BertLayer()
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_attention_heads)])

    def forward(self, hidden_states, all_audio_data, attention_mask=None, output_all_encoded_layers=False):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, all_audio_data, attention_mask=None)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, hidden_size=768):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

def mean_pooling(model_output, attention_mask):
    #token_embeddings = model_output[0]  # last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    return torch.sum(model_output * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class BERT(nn.Module):
    def __init__(self, n_layers=1, d_model=768, model_name="yjernite/retribert-base-uncased",
                 tok_name = "yjernite/retribert-base-uncased"):
        super(BERT, self).__init__()
        #config = AutoConfig.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name)

        self.bert_model = self.model.bert_query

        self.cross_attention1 = BertAttention()
        self.cross_attention2 = BertAttention()

        self.self_attention1 = BertAttention()
        self.self_attention2 = BertAttention()

        self.embed_q = embed_questions_for_retrieval
        self.embed_a = embed_passages_for_retrieval

        self.linear_down = torch.nn.Linear(768 * 3, 144)
        self.linear_up = torch.nn.Linear(144, 768 * 3)

        self.classifier = nn.Linear(768 * 3, 2)
        self.dropout = nn.Dropout(p = 0.1)

        self.act = nn.GELU()

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, question, answer):
        q_tok = self.tokenizer(question, padding = "max_length", max_length = 128,
                               truncation = True, return_tensors= "pt")
        a_tok = self.tokenizer(answer, padding = "max_length", max_length = 128,
                               truncation = True, return_tensors = "pt")

        q_input_ids, q_attention_mask, a_input_ids, a_attention_mask = q_tok['input_ids'].to(device),\
                                                                       q_tok['attention_mask'].to(device), \
                                                                       a_tok['input_ids'].to(device), \
                                                                       a_tok['attention_mask'].to(device)
        #q_output, _ = embed_questions_for_retrieval(question, self.tokenizer, self.model, device = "cuda")
        #a_output, _ = embed_passages_for_retrieval(answer, self.tokenizer, self.model, max_length= 128, device = "cuda")

        q_encoder = self.bert_model(q_input_ids, q_attention_mask)
        a_encoder = self.bert_model(a_input_ids, a_attention_mask)

        last_q = q_encoder['last_hidden_state']
        last_a = a_encoder['last_hidden_state']

        self_q = self.self_attention1(last_q, last_q)
        self_a = self.self_attention2(last_a, last_a)

        cross_q = self.cross_attention1(self_q, self_a)
        cross_a = self.cross_attention2(self_a, self_q)


        pooling_q = cross_q[:, 0, :]
        pooling_a = cross_a[:, 0, :]
        """pooling_q = mean_pooling(cross_q, q_attention_mask)
        pooling_a = mean_pooling(cross_a, a_attention_mask)"""
        pooling_qa = torch.abs(pooling_q - pooling_a)

        #attn_q = self.cross_attention1(q_encoder, a_encoder)
        #attn_a = self.cross_attention2(a_encoder, q_encoder)



        """output_q = self.model.encoder(attn_q)
        output_a = self.model.encoder(attn_a)
        print(output_q['last_hidden_state'].shape)"""
        #q = self.linear1(attn_q[:, 0, :])
        #a = self.linear2(attn_a[:, 0, :])

        #pooled_q = mean_pooling(attn_q, q_attention_mask)
        #pooled_a = mean_pooling(attn_a, a_attention_mask)
        qa_concat = torch.cat((pooling_q, pooling_a, pooling_qa), dim = 1)

        down_output = self.linear_down(self.act(qa_concat))
        up_output = self.linear_up(self.act(down_output))
        #qa_concat = torch.cat((attn_q[:, 0, :], attn_a[:, 0, :]), dim = 1)


        """q_output = self.bert_model(q_input_ids, q_attention_mask)
        a_output = self.bert_model(a_input_ids, a_attention_mask)
        pooled_q = q_output['pooler_output']
        pooled_a = a_output['pooler_output']
        #cross_a = self.cross_attention2(pooled_a, pooled_q)
        #output = cross_a[:, 0, :]"""



        """q_pooled = self.dropout(self.act(mean_pooling(q_output, q_attention_mask)))
        a_pooled = self.dropout(self.act(mean_pooling(a_output, a_attention_mask)))

        qa_dis = torch.abs(q_pooled - a_pooled)
        output = torch.cat((q_pooled, a_pooled, qa_dis), dim = 1)"""


        logits = self.classifier(self.dropout(self.act(up_output)))

        return logits

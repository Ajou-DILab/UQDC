class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.cosim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.dropout = nn.Dropout(p = 0.3)

    def forward_once(self, input_ids, attn_masks):
        model_output = self.model(input_ids = input_ids, attention_mask = attn_masks)
        
        pooled_output = model_output['pooler_output']

        output = self.dropout(nn.functional.gelu(pooled_output))

        return output

    def forward(self, q_input_ids, q_attn_masks, a_input_ids, a_attn_masks):
        
        q_output = self.forward_once(q_input_ids, q_attn_masks)
        a_output = self.forward_once(a_input_ids, a_attn_masks)

        cos = self.cosim(q_output, a_output)

        return cos

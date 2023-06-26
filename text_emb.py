import torch
from diffusers.pipelines.latent_diffusion import LDMBertModel
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from transformers import CLIPTextConfig, CLIPTextModel


class SplitEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_dim):
        super().__init__(vocab_size, embed_dim)
        self.concept_token = Parameter(torch.Tensor(1, embed_dim))

    def forward(self, input: torch.Tensor):
        weight_ = torch.cat([self.weight, self.concept_token])
        return F.embedding(input, weight_, self.padding_idx, self.max_norm,
                           self.norm_type, self.scale_grad_by_freq, self.sparse)


class TextualInversionCLIPTextModel(CLIPTextModel):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        vocab_size, embed_dim = self.text_model.embeddings.token_embedding.weight.size()
        self.text_model.embeddings.token_embedding = SplitEmbedding(vocab_size, embed_dim)

    def patch_emb(self, init_embedding: torch.Tensor):
        self.text_model.embeddings.token_embedding.concept_token = Parameter(init_embedding.unsqueeze(0))
        self.text_model.embeddings.token_embedding.weight.requires_grad = False


class TextualInversionLDMBertModel(LDMBertModel):
    def __init__(self, config):
        super().__init__(config)
        embed_dim = config.d_model
        self.model.embed_tokens = SplitEmbedding(config.vocab_size, embed_dim)

    def patch_emb(self, init_embedding: torch.Tensor):
        self.model.embed_tokens.concept_token = Parameter(init_embedding.unsqueeze(0))
        self.model.embed_tokens.weight.requires_grad = False


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.olmoe.modeling_olmoe import OlmoeMLP
from transformers import AutoTokenizer
import transformers.models.olmoe.modeling_olmoe as modeling_olmoe
import datasets

selected_inds = []

class OlmoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([OlmoeMLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        print("FWD")
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        global selected_inds
        selected_inds.append(selected_experts)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be selected
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

# Override the original olmoe model to use the new OlmoeSparseMoeBlock
modeling_olmoe.OlmoeSparseMoeBlock = OlmoeSparseMoeBlock

def main():
    # load data mix
    dataset = datasets.load_dataset("EleutherAI/wikitext_document_level", name="wikitext-2-v1", split="validation")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
    # tokenize
    req_length = 512
    dataset = dataset.map(lambda x: tokenizer(x["page"], truncation=True, max_length=req_length), batched=True)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) == req_length)
    batch_size = 4
    # create dataloader
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([torch.tensor(x["input_ids"]) for x in batch]),
            "attention_mask": torch.stack([torch.tensor(x["attention_mask"]) for x in batch]),
        }
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)
    # load model
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps" 
    device = torch.device(device)
    model = modeling_olmoe.OlmoeForCausalLM.from_pretrained("allenai/OLMoE-1B-7B-0924", device_map="auto", torch_dtype=torch.float16)
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # forward pass
        with torch.inference_mode():
            model(input_ids, attention_mask=attention_mask)

    global selected_inds
    num_layers = model.config.num_hidden_layers
    num_experts = model.config.num_experts_per_tok 
    num_batches = len(dataloader)
    stacked = torch.stack(selected_inds, dim=0)
    selected_inds = stacked.view((num_batches, num_layers, batch_size, req_length, num_experts))
    torch.save(selected_inds, "selected_experts.pt")
    print("Selected experts saved to selected_experts.pt")


if __name__ == "__main__":
    main()
import jax
import jax.numpy as jnp
import numpy as np
import modula
from data.shakespeare import load_shakespeare
import time

model_path = "results/shakespeare_muon_val_loss_1.221_lipschitz_4.079.npz"

def load_model(model_path, checkpoint=0):
    with np.load(model_path, allow_pickle=True) as data:
        args = data['args'].item()
        print(f"Loading checkpoint {checkpoint} of {len(data.keys()) - 2}")
        weights_dict = data[f'weights_checkpoint_{checkpoint}'].item()
        weights = [0 for _ in weights_dict.keys()]
        for i, w in weights_dict.items():
            weights[i] = jnp.array(w)
    model = modula.compound.GPT(**args)
    return model, weights

@jax.jit
def model_forward(x, weights):
    return model(x, weights)

@jax.jit
def sample_next_token(x, key, temperature):
    logits = model_forward(x, weights)
    last_logits = logits[:, -1, :] / temperature
    key, subkey = jax.random.split(key)
    next_token = jax.random.categorical(subkey, last_logits)
    x = jnp.concatenate([x, next_token[:, None]], axis=1)
    return x, key, next_token

def sample(prompt="", ntokens=20, seed=0, temperature=1.0):
    key = jax.random.PRNGKey(seed)
    x = jnp.array([encode(prompt)], dtype=jnp.int32)
    all_tokens = list(x[0])

    for _ in range(ntokens):
        x, key, next_token = sample_next_token(x, key, temperature)
        all_tokens.append(int(next_token[0]))
    
    return decode(all_tokens)

if __name__ == "__main__":
    model, weights = load_model(model_path)
    encode, decode = load_shakespeare(context_length=256, batch_size=1)["encode"], load_shakespeare(context_length=256, batch_size=1)["decode"]

    prompt = "To be or not to "
    temperatures = [0.2, 0.5, 1.0]
    
    print("Warming up jit...")
    _ = sample(prompt=prompt, ntokens=5, temperature=1.0)
    
    for temperature in temperatures:
        start_time = time.time()
        output = sample(prompt=prompt, ntokens=100, temperature=temperature)
        
        print(f"\nTemperature: {temperature}")
        print(output)
"""Evaluate the saved optimized model with 50 fresh masks per damage level."""
import torch
import rogers_model as m

def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = m.ROOT / 'outputs/rogers2004/tuning/adam'
    model, environment = m.load(output)
    m.evaluate(model, environment, output, trials=50, lesion_seed=1707)

if __name__ == '__main__':
    main()

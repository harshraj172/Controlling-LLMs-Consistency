from pdb import runeval
from arguments import solicit_params
from data import process_data
from load import load_best_model, load_model, load_tokenizer
from train import run_prompt_train, run_train_loop


if __name__ == "__main__":
    args = solicit_params()
    tokenizer = load_tokenizer(args)
    datasets = process_data(args, tokenizer)
    if args.do_train:
        model = load_model(args, tokenizer)
        if args.task == 'soft_prompt':
            run_prompt_train(args, model, datasets)
        elif args.task in ['fine_tune']:
            run_train_loop(args, model, datasets)
    else:
        model = load_model(args, tokenizer)
        if args.task == 'soft_prompt':
            run_prompt_train(args, model, datasets)
        elif args.task in ['fine_tune']:
            run_train_loop(args, model, datasets)
        
        runeval(args, model, datasets['test'])

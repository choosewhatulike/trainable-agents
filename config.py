import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_path", type=str, help="")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--prompt_name", type=str, help="")
    parser.add_argument("--character", type=str, help="", default="Beethoven")
    parser.add_argument("--model_name", type=str, help="", default="gpt-3.5-turbo")
    parser.add_argument("--debug", type=int, default=0, help="")
    parser.add_argument("--max_tokens", type=int, default=None, help="")
    
    parsed_args = parser.parse_args()
    
    return parsed_args


args = parse_arguments()

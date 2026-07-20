from data.c2_decision import load_data as load_c2_data


def load_data(args, split="test"):
    if args.data != "c2_decision":
        raise ValueError(f"Unsupported dataset: {args.data}")
    return load_c2_data(args, split=split)

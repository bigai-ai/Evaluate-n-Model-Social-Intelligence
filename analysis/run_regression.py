import gradient

gradient.train_all_human(mask=[None, None, None, None], filename="4dim-human",start=0)
gradient.train_all_llm(mask=[None, None, None, None], filename="4dim--llms")

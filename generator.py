import torch
import argparse
from transformers import GPT2LMHeadModel, AutoTokenizer

torch.manual_seed(2321)

def loggin_and_print(text):
	print(text)

def text_generate(model, tokenizer, input, max_len):
	input_ids = tokenizer.encode(input)
	gen_ids = model.generate(torch.tensor([input_ids], device='cuda'),
							 ############## for beam search #############
							 # num_beams=5,
							 # no_repeat_ngram_size=2,
							 ############## for greedy search #############
							 top_k = 4,
							 top_p = 0.92,
							 temperature= 0.6,
							 ###############################################
							 max_length=max_len,
							 repetition_penalty=2.0,
							 pad_token_id=tokenizer.pad_token_id,
							 eos_token_id=tokenizer.eos_token_id,
							 bos_token_id=tokenizer.bos_token_id,
							 early_stopping=True,
							 do_sample=True,
							 use_cache=True)
	result_text = tokenizer.decode(gen_ids[0,:].tolist())
	loggin_and_print(result_text)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--max_len', type=int, default=100,
						help="결과물의 길이를 조정합니다.")
	parser.add_argument('--input', type=str, default="당연한",
						help="글의 시작 문장입니다.")
	parser.add_argument('--model_path', type=str, default="./checkpoint/train3_1_lr1.2e-4/KoGPT2_checkpoint_20000",
						help="학습된 결과물을 저장하는 경로입니다.")

	args = parser.parse_args()

	model = GPT2LMHeadModel.from_pretrained(args.model_path)
	tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>',
											  unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
	device = torch.device('cuda')
	model.to(device)
	model.eval()
	text_generate(model=model, tokenizer=tokenizer, input=args.input, max_len=args.max_len)
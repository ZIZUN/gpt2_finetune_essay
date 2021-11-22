import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from util.dataset import essay_dataset

import argparse
import logging
logging.basicConfig(filename='./train.log', level=logging.INFO)

def main(epoch, save_path, data_file_path, batch_size, accum_iter, lr):
	model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
	tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>',
											   unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')

	device = torch.device('cuda')
	model.to(device)
	count = 0

	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [{
		'params': [
			p for n, p in param_optimizer
			if not any(nd in n for nd in no_decay)
		],
		'weight_decay':
			0.01
	}, {
		'params':
			[p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
		'weight_decay':
			0.0
	}]

	dataset = essay_dataset(data_file_path, None, tokenizer)
	print("Read_Dataset ok")
	data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

	learning_rate = lr
	optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
	scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=3000, num_training_steps=50000)

	avg_loss = (0.0, 0.0)

	model.train()

	for epoch in range(epoch):
		for i, data in enumerate(data_loader):
			optimizer.zero_grad()
			data = torch.stack(data)
			data = data.transpose(1,0)
			data = data.to(device)

			outputs = model(data, labels=data)
			loss, logits = outputs[:2]

			loss = loss / accum_iter
			loss.backward()

			if accum_iter == 1:  # not gradient accumulation
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad()
			elif ((i + 1) % accum_iter == 0) or (i + 1 == len(data_loader)):  # gradient accumulation
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad()

			avg_loss = (avg_loss[0] * 0.99 + loss, avg_loss[1] * 0.99 + 1.0)

			if count % 10 == 0:
				for param_group in optimizer.param_groups: # get lr from optimizer
					now_lr = param_group['lr']
				print('{0} epoch, {1} steps, loss = {2:.5f}, avg_loss = {3:.5f}, lr = {4:.8f}'
					  . format(epoch, count, loss, avg_loss[0] / avg_loss[1], now_lr))
				logging.info('{0} epoch, {1} steps, loss = {2:.5f}, avg_loss = {3:.5f}, lr = {4:.8f}'
					  . format(epoch, count, loss, avg_loss[0] / avg_loss[1], now_lr))
			# Generate text
			if (count > 0 and count % 100 == 0):
				model.eval()
				# sent = text_generate(model, tokenizer, None, sent="사랑", text_size=200, temperature=0.7,
				# 					   top_p=0.8, top_k=40)

				input_ids = tokenizer.encode('나는')
				gen_ids = model.generate(torch.tensor([input_ids], device=device),
										 max_length=200,
										 repetition_penalty=2.0,
										 pad_token_id=tokenizer.pad_token_id,
										 eos_token_id=tokenizer.eos_token_id,
										 bos_token_id=tokenizer.bos_token_id,
										 use_cache=True)

				generated = tokenizer.decode(gen_ids[0, :].tolist())
				print(generated)
			# Model save
			if (count > 0 and count % 500 == 0):
				model.save_pretrained(save_path + 'KoGPT2_checkpoint_' + str(count))

			count += 1

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--epoch', type=int, default=20000,
						help="epoch 를 통해서 학습 범위를 조절합니다.")
	parser.add_argument('--save_path', type=str, default='./checkpoint/',
						help="학습 결과를 저장하는 경로입니다.")
	parser.add_argument('--data_file_path', type=str, default='data/train.csv',
						help="학습할 데이터를 불러오는 경로입니다.")
	parser.add_argument('--batch_size', type=int, default=8,
						help="batch_size 를 지정합니다.")
	parser.add_argument('--accum_iter', type=int, default=1,
						help="accumulation step 를 지정합니다.")
	parser.add_argument('--lr', type=int, default=4e-4,
						help="accumulation step 를 지정합니다.")
	args = parser.parse_args()

	main(epoch=args.epoch, save_path=args.save_path, data_file_path=args.data_file_path,
		 batch_size=args.batch_size, accum_iter=args.accum_iter, lr=args.lr)
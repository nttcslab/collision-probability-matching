from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import torch, sys

from modules import trigger_counter

version = "version0.1.0"

def printm(logger, text, file=sys.stdout, timestamp=False):
	import time
	if timestamp:
		now_time = "[%s] " % time.strftime("%Y%m%d-%H:%M:%S")
	else:
		now_time = ""
	print(now_time + str(text), end='\n', file=file)
	print(now_time + str(text), end='\n', file=logger, flush=True)	

class model_base(nn.Module):
	def forward(self, x):
		x = self.layers(x)
		p = self.softmax(x)
		return x, p
	
class AUDNN_model(model_base):
	def __init__(self, len_au, num_classes):
		super(AUDNN_model, self).__init__()
		self.len_au  = len_au
		self.softmax = nn.Softmax(dim=1)			
		self.layers  = nn.Sequential()
		self.layers.add_module("linear0", nn.Linear(len_au, 4608))
		self.layers.add_module("linear1", nn.Linear(4608,   5,  bias=False))

class IUDNN_model(model_base):
	def __init__(self, len_au, num_classes):
		super(IUDNN_model, self).__init__()
		self.len_au  = len_au
		self.softmax = nn.Softmax(dim=1)			
		self.layers  = nn.Sequential()
		self.layers.add_module("linear0", nn.Linear(len_au, 4608))
		self.layers.add_module("relu0",   nn.ReLU())
		self.layers.add_module("linear1", nn.Linear(4608, 4608,  bias=False))
		self.layers.add_module("relu1",   nn.ReLU())
		self.layers.add_module("linear2", nn.Linear(4608,	5,  bias=False))
		
class AUDataset(Dataset):
	def __init__(self, action_units, i_action_units,
				 ratings, i_ratings, num_classes):
		self.action_units   = action_units
		self.i_action_units = i_action_units
		self.ratings		= ratings
		self.i_ratings	  = i_ratings
		self.num_classes	= num_classes
	def __getitem__(self, index):
		i_action_unit = self.i_action_units[index]
		action_unit   = self.action_units[i_action_unit]
		rate		  = int(self.ratings[index]) - 1
		rate_onehot   = np.eye(self.num_classes)[rate]
		rate_onehot   = torch.tensor(rate_onehot, dtype=torch.float32)
		i_rate		= self.i_ratings[index]
		return action_unit, rate, rate_onehot, i_rate
	def __len__(self):
		return len(self.ratings)
			
class AUDataset_sample(AUDataset):
	pass
		

class loaders_class(object):
	pass

class dataloader_class(object):
	def __init__(self, args):
		self.args = args
	def num_outputs(self):
		return self.args.num_classes
	def shuffle_test(self):
		return False

class dataloader_premake_class(dataloader_class):
	def drop_icombs(self):
		raise NotImplementedError
	def pre_make(self, drop_icomb, condition, dataset_label,
				 rts_i, rts_d, aus_i, aus_np):
		raise NotImplementedError
	
class dataloader_postmake_class(dataloader_class):
	def post_make(self, i_ratings_dict, drop_icomb, condition, dataset_label,
				  rts_i, rts_d, aus_i, aus_np):
		raise NotImplementedError
	
class dataloader_premake_whole(dataloader_premake_class):
	def drop_icombs(self):
		return [(list(), list())]
	def num_drops_image(self):
		return 0
	def num_drops_respondent(self):
		return 0
	def pre_make(self, drop_icomb, condition, dataset_label,
				 rts_i, rts_d, aus_i, aus_np):
		i_ratings_dict		  = dict()
		i_ratings_dict["train"] = rts_i[:]
		i_ratings_dict["test"]  = rts_i[:]
		return i_ratings_dict

		
class dataloader_postmake_action_unit(dataloader_postmake_class):
	def num_inputs(self):
		return self.args.len_actions
	def post_make(self, i_ratings_dict, drop_icomb, condition, dataset_label,
				  rts_i, rts_d, aus_i, aus_np):

		batch_sizes = \
			{"train":self.batch_size_train(), "test":self.batch_size_test()}
		shuffles	= \
			{"train":self.shuffle_train(),	"test":self.shuffle_test()}

		num_outputs   = self.num_outputs()
		dataset_class = eval("AUDataset_%s" % dataset_label)
		dataloaders   = loaders_class()
		for k, i_ratings in i_ratings_dict.items():
			ratings		= list()
			i_action_units = list()
			batch_size	 = batch_sizes[k]
			shuffle		= shuffles[k]
			for i_rating in i_ratings:
				i_action_unit = [n for n, v in enumerate(aus_i) if v in i_rating[1]]
				assert(len(i_action_unit) == 1)
				i_action_units += i_action_unit
				ratings.append(rts_d[condition][i_rating])
			dataset	= dataset_class(aus_np, i_action_units, ratings, i_ratings,
									   num_outputs)
			dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
									shuffle=shuffle,
									num_workers=self.args.num_workers)
			setattr(dataloaders, k, dataloader)
			
		return dataloaders
			
class dataloader_postmake_unique_action_unit(dataloader_postmake_class):
	def num_inputs(self):
		return self.args.len_actions + self.args.num_respondents
	def post_make(self, i_ratings_dict, drop_icomb, condition, dataset_label,
				  rts_i, rts_d, aus_i, aus_np):
		
		batch_sizes = \
			{"train":self.batch_size_train(), "test":self.batch_size_test()}
		shuffles	= \
			{"train":self.shuffle_train(),	"test":self.shuffle_test()}
		
		dataset_class   = eval("AUDataset_%s" % dataset_label)
		num_respondents = self.args.num_respondents
		min_id		  = self.args.min_id
		num_outputs	 = self.num_outputs()
		dataloaders	 = loaders_class()
		for k, i_ratings in i_ratings_dict.items():
			ratings		= list()
			i_action_units = list()
			action_units   = list()
			batch_size	 = batch_sizes[k]
			shuffle		= shuffles[k]
			for i, i_rating in enumerate(i_ratings):
				i_action_unit = [n for n, v in enumerate(aus_i) if v in i_rating[1]]
				assert(len(i_action_unit) == 1)
				i_action_unit = i_action_unit[0]
				id_respondent = i_rating[0] - min_id
				id_onehot	 = np.eye(num_respondents)[id_respondent]
				action_units.append(np.concatenate([aus_np[i_action_unit],
													id_onehot]))
				i_action_units.append(i)
				ratings.append(rts_d[condition][i_rating])
			c_aus_np   = np.stack(action_units)
			dataset	= dataset_class(c_aus_np, i_action_units, ratings, i_ratings,
									   num_outputs)
			dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
									shuffle=shuffle,
									num_workers=self.args.num_workers)
			setattr(dataloaders, k, dataloader)
			
		return dataloaders

class dataloader_make_pre_post_make(dataloader_class):
	def make(self, drop_icomb, condition, dataset_label,
			 rts_i, rts_d, aus_i, aus_np):
		i_ratings_dict = self.pre_make(drop_icomb, condition, dataset_label,
									   rts_i, rts_d, aus_i, aus_np)
		dataloaders	= self.post_make(i_ratings_dict, drop_icomb, condition,
										dataset_label, rts_i, rts_d, aus_i, aus_np)
		return dataloaders
	
class dataloader_base(dataloader_postmake_action_unit,
					  dataloader_make_pre_post_make):
	def batch_size_train(self):
		return self.args.batch_size_train
	def batch_size_test(self):
		return self.args.num_respondents
	def shuffle_train(self):
		return True

	## for calculating measures for CPM
	def criterion(self, x, rate_onehot):
		ccp	  = (x * rate_onehot).sum(axis=1)
		cp	   = (x * x).sum(axis=1)
		conf	 = x.max(dim=1).values
		
		loss	 = ((ccp - cp) * (ccp - cp)).sum() / x.shape[0]
		ccp_avg  = ccp.sum()  / x.shape[0]
		cp_avg   = cp.sum()   / x.shape[0]
		conf_avg = conf.sum() / x.shape[0]
		
		return loss, ccp_avg, cp_avg, conf_avg

class dataloader_criterion_per_respondent(dataloader_class):
	def shuffle_train(self):
		return False
	def criterion(self, x, rate_onehot):
		num_images	  = self.args.num_images	  - self.num_drops_image()
		num_respondents = self.args.num_respondents - self.num_drops_respondent()
		
		ccp  = (x * rate_onehot).sum(axis=1)
		cp   = (x * x).sum(axis=1)
		conf = x.max(dim=1).values
		
		ccp  = torch.reshape(ccp, (num_respondents, num_images)).sum(axis=1) \
			/ num_images
		cp   = torch.reshape(cp,  (num_respondents, num_images)).sum(axis=1) \
			/ num_images
		conf = torch.reshape(conf,(num_respondents, num_images)).sum(axis=1) \
			/ num_images
		
		loss	 = ((ccp - cp) * (ccp - cp)).sum() / num_respondents
		ccp_avg  = ccp.sum()  / num_respondents
		cp_avg   = cp.sum()   / num_respondents
		conf_avg = conf.sum() / num_respondents

		return loss, ccp_avg, cp_avg, conf_avg
	
class dataloader_unique_action_unit_whole(
		dataloader_postmake_unique_action_unit,
		dataloader_premake_whole,
		dataloader_make_pre_post_make,
		dataloader_criterion_per_respondent):
	def batch_size_train(self):
		return self.args.num_respondents * self.args.num_images
	def batch_size_test(self):
		return self.args.num_respondents * self.args.num_images


def make_dataloaders(args):
	import pandas as pd
	import os, sys, math
	
	if not os.path.exists(args.features):
		printm(args.log,
			   "could not find featire file: %s" % args.features,
			   file=sys.stderr)
		sys.exit(1)
	try:
		aus = pd.read_csv(args.features)
	except Exception as e:
		printm(args.log, e, file=sys.stderr)
		sys.exit(1)
		
	args.len_actions = sum(["AU" in v for v in aus.columns])
	
	aus		= aus.loc[:, aus.columns[0:2].append(aus.columns[-args.len_actions:])]
	try:
		aus_img	= list(aus["img"])
		aus_img_id = list(aus["stim_id"])
	except Exception as e:
		printm(args.log, e, file=sys.stderr)
		printm(args.log, "action-unit file must contain \"img\", \"stim_id\" "
			   "columns", file=sys.stderr)
		sys.exit(1)
	aus_np	 = aus.iloc[:, 2:].to_numpy(dtype=float)
	img_table  = {aus_img[i]:int(-1 if math.isnan(aus_img_id[i]) \
								 else aus_img_id[i]) for i in range(len(aus_img))}
	assert(len(aus_img) == len(set(aus_img)))

	if not os.path.exists(args.inputs):
		printm(args.log,
			   "could not find ratings file: %s" % args.inputs, file=sys.stderr)
		sys.exit(1)
	try:
		rts = pd.read_csv(args.inputs)
	except Exception as e:
		printm(args.log, e, file=sys.stderr)
		sys.exit(1)
	try:
		rts = rts.pivot_table(index=("subj_id", "image"), columns="condition",
							  values="response")
	except Exception as e:
		printm(args.log, e, file=sys.stderr)
		printm(args.log, "rating file must contain \"subj_id\", \"image\", "
			   "\"condition\", \"response\" columns", file=sys.stderr)
		sys.exit(1)
	rts_i = [(v[0], os.path.basename(v[1]))   for v in rts.index.to_list()]
	rts_d = rts.to_dict()

	
	rts_d = {c:{(k[0], os.path.basename(k[1])):v \
				for k, v in rts_d[c].items()} for c in rts_d.keys()}

	crts_i = list()
	ids	= list(set([v[0] for v in rts_i]))
	imgs   = list(set([v[1] for v in rts_i]))
	ids.sort()
	imgs.sort()
	for i in range(len(ids) - 1):
		assert(ids[i + 1] - ids[i] == 1)
	for id in ids:
		for img in imgs:
			key = (id, img)
			assert(key in rts_i)
			crts_i.append(key)
	rts_i = crts_i			

	args.img_table		= img_table
	args.make_dataloader  = eval("dataloader_%s" % args.loader_type)	
	args.make_dataloader  = args.make_dataloader(args)		
	args.num_images	   = len(set(rts.reset_index()[:]["image"]))
	args.num_respondents  = len(set(rts.reset_index()[:]["subj_id"]))
	args.min_id		   = min(ids)
	args.num_outputs	  = args.make_dataloader.num_outputs()
	args.num_inputs	   = args.make_dataloader.num_inputs()
	args.batch_size_train = args.make_dataloader.batch_size_train()
	args.batch_size_test  = args.make_dataloader.batch_size_test()
	args.shuffle_train	= args.make_dataloader.shuffle_train()
	args.shuffle_test	 = args.make_dataloader.shuffle_test()
	args.drop_icombs	  = args.make_dataloader.drop_icombs()
	args.rts_i			= rts_i
	args.rts_d			= rts_d
	args.aus_img		  = aus_img
	args.aus_np		   = aus_np
	args.num_buffers	  = len(args.drop_icombs) \
		if args.num_buffers is None else args.num_buffers
	

def save_pxouts(args, xout, pout, rates, i_rate,
				cv_label, epoch, case_label):
	import os, csv
	
	responses	  = rates.tolist()
	respondent_ids = i_rate[0].tolist()	
	for label, v in [["xout", xout], ["pout", pout]]:
		pxouts = v.tolist()
		outs   = list()
		for i in range(len(pxouts)):
			img_name	  = i_rate[1][i]
			img_id		= args.img_table[img_name]
			pxout		 = pxouts[i]
			response	  = responses[i] + 1
			respondent_id = respondent_ids[i]
			out		   = [img_id, respondent_id, img_name] + pxout + [response]
			outs.append(out)
		d_file = (cv_label + "_e%d_distribution%s_%s.csv") % \
			(epoch + 1, case_label, label)
		d_path = os.path.join(args.pxout_path, d_file)
		with open(d_path, "w") as f:
			f.write(",".join(["img_id", "subj_id", "image",
							  "V1", "V2", "V3", "V4", "V5", "response"]) + "\n")
			csv.writer(f).writerows(outs)


def select_scheduler(args, optimizer):
	from torch.optim.lr_scheduler import MultiStepLR
	
	class scheduler_none(object):
		def __init__(self, lr):
			self.learning_rate = lr
		def step(self):
			pass
		def get_last_lr(self):
			return [self.learning_rate]

	if args.scheduler is None:
		scheduler = scheduler_none(args.optimizer_parameters["lr"])
		return scheduler
	
	if args.scheduler == "MultiStepLR":
		milestones = [(0.5 + i / 10) * args.num_epochs for i in range(5)]
		args.scheduler_parameters["milestones"] = milestones

	try:
		scheduler = eval("%s" % args.scheduler)
		scheduler = scheduler(optimizer, **args.scheduler_parameters)
	except Exception as e:
		printm(args.log, e, file=sys.stderr)
		printm(args.log, "there is argument which is not supported by this scheduler",
			   file=sys.stderr)
		sys.exit(1)

	states	   = scheduler.state_dict()
	keys		 = list(states.keys())
	state_format = " %-{}s: %s".format(max([len(k) for k in keys]))
	keys.sort()
	printm(args.log, "scheduler informations")
	for k in keys:
		printm(args.log, state_format % (k, states[k]))
	printm(args.log, "")
	
	return scheduler


def initialize_weights(args, model):
	if args.init_weights is None:
		return
	init = eval("nn.init.%s" % args.init_weights)
	for layer in model.layers:
		if not hasattr(layer, "weight"):
			continue
		init(layer.weight)

		
def training_condition(args, dataloaders):
	from decimal import Decimal
	import pandas as pd
	import sys, time, os

	class cost_class(object):
		def __init__(self, cost_labels):
			self.costs = dict()
			for cost_label in cost_labels + ["cost"]:
				self.costs[cost_label] = list()
		
	num_inputs	   = args.make_dataloader.num_inputs()
	num_outputs	  = args.make_dataloader.num_outputs()
	batch_size_train = args.make_dataloader.batch_size_train()
	batch_size_test  = args.make_dataloader.batch_size_test()
	model_class	  = eval("%s_model" % args.model)
	
	costs		= list()
	model		= model_class(num_inputs, num_outputs)
	initialize_weights(args, model)
	model.to(args.cuda_device)
	
	model_keys   = model._modules.keys()
	model_format = " %-{}s: %s".format(max([len(k) for k in model_keys]))
	printm(args.log, "model informations")
	for k in model_keys:
		printm(args.log, model_format % (k, str(model._modules[k])))
	printm(args.log, "")

	try:
		optimizer = args.optimizer(model.parameters(), **args.optimizer_parameters)
	except Exception as e:
		printm(args.log, str(e), file=sys.stderr)
		printm(args.log, "there is argument which is not supported by this optimizer",
			   file=sys.stderr)
		sys.exit(1)

	printm(args.log, "optimizer informations")
	printm(args.log, optimizer)
	printm(args.log, "")
	
	scheduler	   = select_scheduler(args, optimizer)
	accuracies	  = list()
	predict_times   = torch.arange(1)
	criterion_cross = nn.CrossEntropyLoss()
	cv_format	   = "%0{}dcv_%s".format(len(str(len(dataloaders))))		
	
	for cv_epoch, (train_loader, test_loader) in enumerate(dataloaders):
		cv_epoch += args.cv_epoch_threshold
		printm(args.log_train,
			   str(cv_epoch) + "-th train. %s. input(%d,%d)->output(%d,%d)" % \
			   (dataloaders.name,
				batch_size_train, num_inputs,
				batch_size_train, num_outputs), timestamp=True)
		cv_label = cv_format % (cv_epoch, dataloaders.name)
		for epoch in range(args.num_epochs):
			train_num	  = 0
			train_true_num = 0
			count		  = 0
			ccps		   = list()
			cps			= list()
			confs		  = list()
			diffs		  = list()
			costs		  = cost_class(args.cost_labels)
			model.train()
			for batch_idx, (action_unit, rate, rate_onehot, i_rate) \
				in enumerate(train_loader):
				action_unit = action_unit.to(args.cuda_device)
				rate		= rate.to(args.cuda_device)
				rate_onehot = rate_onehot.to(args.cuda_device)
				xout, pout  = model(action_unit.float())
				
				loss, ccp, cp, conf =\
					args.make_dataloader.criterion(pout, rate_onehot)
				
				ipout		   = pout.argmax(1)
				train_num	  += rate.shape[0]
				train_true_num += (ipout == rate).sum()
				accuracy		= train_true_num / train_num

				cost1 = loss * args.weight_loss
				cost2 = criterion_cross(xout, rate)
				cost  = eval("+".join(args.cost_labels))
				
				optimizer.zero_grad()
				cost.backward()
				optimizer.step()

				confs.append(conf.tolist())
				ccps.append(ccp.tolist())
				cps.append(cp.tolist())
				diffs.append(loss.tolist())
				for cost_label in args.cost_labels + ["cost"]:
					eval("costs.costs['%s'].append(%s.tolist())" \
						 % (cost_label, cost_label))
				count += 1

			scheduler.step()
			accuracy_train = train_true_num / train_num

			cost_log = " | ".join(["%s:%.4f" % (cost_label, \
												np.mean(costs.costs[cost_label]))\
								   for cost_label in ["cost"] + args.cost_labels])
			
			printm(args.log_train,
				   "train | count:%d | epoch:%03d/%03d | lr:%e | accuracy:%.4f | "
				   "ccp:%.4f | cp:%.4f | diff:%e | conf:%.4f | %s" % \
				   (count, epoch + 1, args.num_epochs, scheduler.get_last_lr()[0],
					accuracy_train,
					np.mean(ccps), np.mean(cps), np.mean(diffs), np.mean(confs),
					cost_log))

			mean_ccp = np.mean(ccps)
			mean_cp  = np.mean(cps)
			if Decimal(str(mean_ccp)).quantize(Decimal('0.0000')) == \
			   Decimal(str(mean_cp)).quantize(Decimal('0.0000')):  
				printm(args.log_train,
					   'CCP and hatCP are matched: %.4f' %  np.mean(ccps))

			if args.epoch_counter.check():
				if count != 1:
					printm(args.log, "this dataloader does not support save pxouts",
						   sys.stderr)
					sys.exit(1)
				save_pxouts(args, xout, pout, rate, i_rate,
							cv_label, epoch, "")
				
		model_file = cv_label + "_AU.pth"
		model_path = os.path.join(args.model_path, model_file)
		torch.save(model.state_dict(), model_path)

		q	 = 0
		model = model.eval()
		count = 0
		model.to(args.cuda_device)

		printm(args.log_train,
			   str(cv_epoch) + "-th test. %s. input(%d,%d)->output(%d,%d)" % \
			   (dataloaders.name,
				batch_size_test, num_inputs, batch_size_test, num_outputs),
			   timestamp=True)

		del xout, pout, rate, rate_onehot, i_rate
		while q <= int(torch.max(predict_times)):
			test_num	  = 0
			test_true_num = 0
			q			+= 1
			for batch_idx, (action_unit, rate, rate_onehot, i_rate) \
				in enumerate(test_loader):
				action_unit = action_unit.to(args.cuda_device)
				rate		= rate.to(args.cuda_device)
				rate_onehot = rate_onehot.to(args.cuda_device)
				xout, pout  = model(action_unit.float())

				pred		   = pout.argmax(1)
				test_num	  += rate.shape[0]
				test_true_num += (pred == rate).sum().item()
				count		 += 1

			accuracy_test = test_true_num / test_num
			printm(args.log_train,
				   "test | count:%d | cv:%03d/%03d | accuracy:%.4f" %\
				   (count, cv_epoch + 1, len(args.drop_icombs), accuracy_test))

			if args.epoch_counter.active():
				if count != 1:
					printm(args.log, "this dataloader does not support save pxouts",
						   sys.stderr)
					sys.exit(1)
				save_pxouts(args, xout, pout, rate, i_rate,
							cv_label, epoch, "_test")

		accuracies.append((accuracy_train, accuracy_test))

		del model, optimizer
		model	 = model_class(num_inputs, num_outputs)
		initialize_weights(args, model)
		model.to(args.cuda_device)
		optimizer = args.optimizer(model.parameters(), **args.optimizer_parameters)

	accuracies_file = "AU_%s_onehot_accuracy_training_test.csv" % dataloaders.name
	accuracies_path = os.path.join(args.output_dir, accuracies_file)
	pd.DataFrame(accuracies).to_csv(accuracies_path)

	
def training(args):
	from joblib import Parallel, delayed
	import numpy as np
	import math, os

	class dataloaders_base(object):
		def __init__(self, loaders):
			self.trains = [v.train	  for v in loaders]
			self.tests  = [v.test	   for v in loaders]
		def __getitem__(self, i):
			return self.trains[i], self.tests[i]
		def __len__(self):
			return len(self.trains)

	class dataloaders_sample(dataloaders_base):
		@property
		def name(self):
			return "sample"


	for dataset_label, condition in [("sample", 3)]:
		if not dataset_label in args.conditions:
			continue
		log_file				= "cv_onehot_%s.log" % dataset_label
		args.log_train		  = open(os.path.join(args.output_dir, log_file),
									   mode='w')
		args.cv_epoch_threshold = 0
		num_splits			  = math.ceil(len(args.drop_icombs) / args.num_buffers)
		drop_icombs_np		  = np.array(args.drop_icombs, dtype=object)
		for split_drop_icombs in np.array_split(drop_icombs_np, num_splits):
			split_drop_icombs = split_drop_icombs.tolist()
			log			   = args.log
			args.log		  = None
			log_train		 = args.log_train
			args.log_train	= None
			tmp = Parallel(n_jobs=args.num_jobs_torch, backend="multiprocessing")\
				(delayed(args.make_dataloader.make)(drop_icomb, condition,
													dataset_label,
													args.rts_i, args.rts_d,
													args.aus_img, args.aus_np) \
				 for drop_icomb in split_drop_icombs)
			dataloaders	= eval("dataloaders_%s" % dataset_label)(tmp)
			args.log	   = log
			args.log_train = log_train
			training_condition(args, dataloaders)
			args.cv_epoch_threshold += len(split_drop_icombs)
			
		
def main():
	import argparse, sys, os, torch, inspect

	class argparse_parameter(argparse.Action):
		def __call__(self, parser, namespace, values, option_string=None):
			params = dict()
			for value in values:
				key2value = value.split(":")
				if len(key2value) != 2:
					print("error occured at %s" % self.dest, file=sys.stderr)
					print("unproper argument is found: %s" % value, file=sys.stderr)
					print("please set format key:value (<- no space!)", file=sys.stderr)
					sys.exit(1)
				key = key2value[0]
				try:
					num = float(key2value[1])
				except Exception as e:
					print("error occured at %s" % self.dest, file=sys.stderr)
					print(e, file=sys.stderr)
					print("could not convert to float: %s" % key2value[1], file=sys.stderr)
					sys.exit(1)
				params[key] = num
			setattr(namespace, self.dest, params)
				
	parser = argparse.ArgumentParser(
		description = "Example code of collision probability matching"
		)
	parser.add_argument("--version", "-v",
						action="version", version=version,
						help="show version information"
						)
	parser.add_argument("--featires", "-a", dest="features",
						type=str, nargs=1,
						default=["data/features.csv"],
						help="feature file path"
						)
	parser.add_argument("--output", "-o", dest="output_dir",
						type=str, default=["output"], nargs=1,
						help="output directory"
						)
	parser.add_argument("--inputs", "-r", dest="inputs",
						type=str, nargs=1,
						default=["data/labels.csv"],
						help="ratings file path"
						)
	parser.add_argument("--num_classes", "-c",
						type=int, default=[5], nargs=1,
						help="number of classes"
						)
	parser.add_argument("--cuda_index",
						type=int, default=[0], nargs=1,
						help="cuda index"
						)
	parser.add_argument("--num_workers",
						type=int, default=[3], nargs=1,
						help="number of workers for data loader"
						)
	parser.add_argument("--num_epochs",
						type=int, default=[100], nargs=1,
						help="number of epochs for learning process"
						)
	parser.add_argument("--batch_size", dest="batch_size_train",
						type=int, default=[60], nargs=1,
						help="batch size for learning process"
						)
	parser.add_argument("--learning_rate", "-l",
						type=float, default=[2e-04], nargs=1,
						help="learning rate for learning process"
						)
	parser.add_argument("--n_jobs", dest="num_jobs_torch",
						type=int, default=[-1], nargs=1,
						help="number of jobs for making dataloaders"
						)
	parser.add_argument("--optimizer",
						type=str, default=["Adam"], nargs=1,
						choices=["Adam", "Adadelta", "Adagrad", "AdamW", "SparseAdam",
								 "Adamax", "ASGD", "LBFGS", "NAdam", "RMSprop", "Rprop",
								 "SGD"],
						help="optimizer"
						)
	parser.add_argument("--optimizer_parameters",
						type=str, action=argparse_parameter,
						nargs="+",
						help="arguments for optimizer. format key:value (<- no space!)"
						)	
	parser.add_argument("--seed", dest="seed_torch",
						type=int, default=[None], nargs=1,
						help="seed number of torch"
						)
	parser.add_argument("--weight", "-w", dest="weight_loss",
						type=float, default=[100000], nargs=1,
						help="weight value for loss value"
						)
	parser.add_argument("--weight2", "-w2", dest="weight_loss2",
						type=float, default=[1], nargs=1,
						help="weight value for accuracy loss"
						)
	parser.add_argument("--dataloader_type", dest="loader_type",
						type=str, default=["unique_action_unit_whole"], nargs=1,
						choices=["base",
								 "unique_action_unit_whole"
						],
						help="dataloader type name"
						)
	parser.add_argument("--costs", dest="cost_labels",
						type=str, default=["cost1", "cost2"], nargs="+",
						choices=["cost1", "cost2"],
						help="select cost label to calculate cost value"
						)
	parser.add_argument("--num_epochs_at_xpout", 
						type=int, default=[100], nargs=1,
						help="number of epochs at output xpout files"
						)
	parser.add_argument("--conditions",
						type=str, default=["sample"], nargs="+",
						choices=["sample"],
						help="condition labels for process"
						)
	parser.add_argument("--num_buffers",
						type=int, default=[None], nargs=1,
						help="number of buffers for multithreading"
						)
	parser.add_argument("--model", "-m",
						type=str, choices=["AUDNN", "IUDNN"], nargs=1,
						default=["AUDNN"],
						help="neural network model"
						)
	parser.add_argument("--scheduler",
						type=str,
						choices=["MultiStepLR", "StepLR", "ConstantLR",
								 "LinearLR", "ExponentialLR", "CosineAnnealingLR",
								 "ReduceLROnPlateau", "CyclicLR", "OneCycleLR",
								 "CosineAnnealingWarmRestarts"],
						nargs=1, default=[None],
						help="scheduler for training step"
						)
	parser.add_argument("--scheduler_parameters",
						type=str, action=argparse_parameter,
						nargs="+",
						help="arguments for scheduler. format key:value (<- no space!)"
						)	
	parser.add_argument("--init_weights",
						type=str,
						choices=["uniform_", "normal_", "ones_", "zeros_",
								 "eye_", "dirac_", "xavier_uniform_", "xavier_normal_",
								 "kaiming_uniform_", "kaiming_normal_", "trunc_normal_",
								 "orthogonal_"],
						nargs=1, default=[None],
						help="method to initialize weights"
						)
	args = parser.parse_args()

	for k in args.__dict__.keys():
		if k in ["cost_labels", #"conditions",
				 "optimizer_parameters", "scheduler_parameters"]:
			continue
		setattr(args, k, getattr(args, k)[0])

	if args.optimizer_parameters is None:
		args.optimizer_parameters = dict()
	if args.scheduler_parameters is None:
		args.scheduler_parameters = dict()
		
	if not hasattr(args.optimizer_parameters, "lr"):
		args.optimizer_parameters["lr"] = args.learning_rate
	args.learning_rate = None # <- just assert
		
	if not args.seed_torch is None:
		torch.manual_seed(args.seed_torch)
		torch.cuda.manual_seed(args.seed_torch)
		
	os.makedirs(args.output_dir, exist_ok=True)

	args.log			 = open(os.path.join(args.output_dir, "log.log"), mode='w')
	args.torch_version   = torch.__version__
	args.version		 = version
	args.cuda_device	 = "cuda:%d" % args.cuda_index
	args.model_path	  = os.path.join(args.output_dir, "baseline") 
	args.pxout_path	  = os.path.join(args.output_dir, "pxouts") 
	args.dataloaders	 = make_dataloaders(args)

	args_keys  = list(args.__dict__.keys())
	log_format = " %-{}s: %s".format(max([len(k) for k in args_keys]))
	args_keys.sort()
	printm(args.log, "%s" % sys.argv[0])
	printm(args.log, "\t\t%s\n" % args.version)
	printm(args.log, "program arguments information")
	for k in args_keys:
		if k in ["log", "version", "dataloaders", "make_dataloader", "min_id",
				 "img_table", "drop_icombs", "rts_i", "rts_d", "aus_img", "aus_np",
				 "learning_rate"]:
			continue
		printm(args.log, log_format % (k, str(getattr(args, k))))

	methods	   = {n[1].__func__.__name__ : n[1] for n in \
					 inspect.getmembers(args.make_dataloader, inspect.ismethod)}
	method_keys   = list(methods.keys())
	method_format = " %-{}s: %s".format(max([len(n) for n in methods.keys()]))
	method_keys.sort()
	printm(args.log, "\nmethod informations")
	for name in method_keys:
		method = methods[name]
		printm(args.log, method_format % (name, method))

	args.optimizer	 = eval("torch.optim.%s" % args.optimizer)
	args.epoch_counter = trigger_counter.trigger_counter(args.num_epochs_at_xpout)
	
	os.makedirs(args.model_path, exist_ok=True)
	os.makedirs(args.pxout_path, exist_ok=True)
	training(args)


if __name__ == '__main__':
	main()

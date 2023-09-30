import os
import json
import ast
import math
import sys

import pandas as pd
import numpy as np
from sklearn import linear_model, metrics

import pdb

max_list_size = 50

# single situation
situations = range(1, 46)

def get_ndcg(click_list):
	num_click = 0
	for r in range(max_list_size):
		if click_list[r] == 1:
			num_click += 1

	if num_click == 0:
		# print("{}: {}".format(click_list, 0))
		return 0

	indcg = 0
	if num_click == 1:
		indcg = 1
	else:
		for r in range(num_click):
			indcg += (1 / math.log(r + 2, 2))

	ndcg = 0
	for r in range(max_list_size):
		ndcg += (click_list[r] / math.log(r + 2, 2))

	# print("{}: {}".format(click_list, ndcg / indcg))

	return (ndcg / indcg)


def get_rewarding_scores(infolder):
	with open(os.path.join(infolder, 'predict_meta.txt')) as f:
		predict_data = f.read()
	predict_meta = ast.literal_eval(predict_data)

	predict_dataset = []
	lines = open(os.path.join(infolder, 'predict_per_query_quid.txt')).readlines()
	for line in lines:
		attr = line.strip().split('\t')
		sid = int(attr[0].strip())
		qid = int(attr[1].strip())
		pids = json.loads(attr[2].strip())
		scores = json.loads(attr[3].strip())
		predict_dataset.append({
			'sid': sid,
			'qid': qid,
			'pids': pids,
			'scores': scores
		})

	predict_results = []
	lines = open(os.path.join(infolder, 'predict_results.txt')).readlines()
	for line in lines:
		attr = line.strip().split(' ')
		predict_results.append([float(fl) for fl in attr])


	rewarding_scores = {}

	for i, predict_item in enumerate(predict_dataset):
		this_pids = predict_dataset[i]["pids"]
		this_pred = predict_results[i]

		first_zero_pos = 0
		while this_pids[first_zero_pos] > 0: first_zero_pos += 1
		this_pids = this_pids[0:first_zero_pos]
		this_pred = this_pred[0:first_zero_pos]

		this_pred_normalized = [pr / sum(this_pred) for pr in this_pred]

		predict_item_meta = predict_meta[i]

		if predict_item_meta[0][1] != 0:
			split = 0.5 ## predict_item_meta[0] has multiple risks, thus a split is needed
		else:
			split = 1

		cat_id = predict_item_meta[1]
		support = predict_item_meta[2]
		for risk_id in predict_item_meta[0]:
			if risk_id == 0:
				continue

			if risk_id not in rewarding_scores:
				rewarding_scores[risk_id] = {}
			if cat_id not in rewarding_scores[risk_id]:
				rewarding_scores[risk_id][cat_id] = {}

			for j, pid in enumerate(this_pids):
				if pid not in rewarding_scores[risk_id][cat_id]:
					rewarding_scores[risk_id][cat_id][pid] = 0
				rewarding_scores[risk_id][cat_id][pid] += split * support * this_pred_normalized[j]

	df_prd_info = pd.read_csv(os.path.join(infolder, 'prd_info.csv'))
	df_cri_info = pd.read_csv(os.path.join(infolder, 'cri_info.csv'))

	return rewarding_scores, df_prd_info, df_cri_info


def run(updating_rate, rewarding_scores, df_prd_info, df_cri_info, infolder):

	'''
	Get updated weights
	'''
	updated_results = {}
	for risk_id in rewarding_scores:
		updated_results[risk_id] = {}

		for cat_id in rewarding_scores[risk_id]:
			updated_results[risk_id][cat_id] = {}

			rewarding_scores_risk_cat = rewarding_scores[risk_id][cat_id]
			# print(f"risk = {risk_id}, cat = {cat_id}")
			
			rel_df = df_prd_info[(df_prd_info["situation_id"] == risk_id) & (df_prd_info["category_id"] == cat_id)]

			rel_prd_list = rel_df["product_id"].unique()

			rel_prd_collected = {}

			reg_X = []
			reg_y = []

			weights = []
			weights_normalized = []

			first_itr = True

			for rel_prd in rel_prd_list:
				rel_df_prd = rel_df[(rel_df["product_id"] == rel_prd)]
				n_dim = len(rel_df_prd)

				prd_attr_normalized = []
				for i in range(n_dim):
					item_prd = rel_df_prd.iloc[i]

					if first_itr:
						weights.append(item_prd["weight"] * item_prd["highlight_factor"])

					item_info = df_cri_info[(df_cri_info["term_cat_id"] == item_prd["category_id"]) & (df_cri_info["term_verbose_code"] == item_prd["term_verbose_code"])]
					
					assert(len(item_info) == 1)
					assert(item_info.iloc[0]["full_mark"] != 0)

					prd_attr_normalized.append(item_prd["term_sum_score"] / item_info.iloc[0]["full_mark"])
					# prd_attr_normalized.append(item_prd["term_sum_score"])
					### note that there is a problem here
					### if prd_attr is normalized, then the weights are changed, so the result will not equal to the current list
					### workaround: not normalize this score !!! this has been fixed

				if first_itr:
					weights_normalized = [w / sum(weights) for w in weights]

				score_normalized = 0
				for i in range(n_dim):
					score_normalized += weights_normalized[i] * prd_attr_normalized[i]

				score_rewarding = rewarding_scores_risk_cat[rel_prd]

				rel_prd_collected[rel_prd] = {
					"attr": prd_attr_normalized,
					# "w_curr": prd_weight_normalized,
					"s_curr": score_normalized,
					"s_rew": score_rewarding,
					"s_exp": score_normalized + updating_rate * 2000.0 * score_rewarding ## for a normalizing
				}

				# print("product:", rel_prd)
				# print("{} -> {}".format(rel_prd_collected[rel_prd]["s_curr"], rel_prd_collected[rel_prd]["s_exp"]))

				reg_X.append(rel_prd_collected[rel_prd]["attr"])
				reg_y.append(rel_prd_collected[rel_prd]["s_exp"])

				first_itr = False

			# print("current weights normalized:", weights_normalized)

			reg_X = np.array(reg_X)
			reg_y = np.array(reg_y)

			### Note that the two parameter setting for LinearRegression are necessary and important
			### Without the first, negative weights could be computed
			### Without the second, the model seem to compute a constant, but our data is "centered"
			### thus fit_intercept=False is needed according to:
			### https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
			reg = linear_model.LinearRegression(positive=True, fit_intercept=False).fit(reg_X, reg_y)
			  
			# # regression coefficients
			# print('Coefficients: ', reg.coef_)
			  
			# # variance score: 1 means perfect prediction
			# print('Variance score: {}'.format(reg.score(reg_X, reg_y)))

			new_weights_normalized = [c / sum(reg.coef_) for c in reg.coef_]
			# print("new weights normalized:", new_weights_normalized)

			# sort_check = [(p, rel_prd_collected[p]["s_curr"]) for p in rel_prd_collected]
			# sort_check.sort(key=lambda a: a[1], reverse=True)
			# print("current list:", [a[0] for a in sort_check])

			for rel_prd in rel_prd_collected:
				new_score = 0
				for i, attr in enumerate(rel_prd_collected[rel_prd]["attr"]):
					new_score += new_weights_normalized[i] * attr
				rel_prd_collected[rel_prd]["s_new"] = new_score

			# sort_check = [(p, rel_prd_collected[p]["s_new"]) for p in rel_prd_collected]
			# sort_check.sort(key=lambda a: a[1], reverse=True)
			# print("current list:", [a[0] for a in sort_check])

			updated_results[risk_id][cat_id]["rel_prd"] = rel_prd_collected
			updated_results[risk_id][cat_id]["old_wts"] = weights_normalized
			updated_results[risk_id][cat_id]["new_wts"] = new_weights_normalized
	'''
	Get updated weights
	'''

	'''
	Write updated weights results
	'''
	with open(os.path.join(infolder, 'updated_weights-' + str(updating_rate) + '.csv'), "w", newline='') as outfile:
		cat_ids = [2, 4, 5, 6]
		df_aggr_info = df_prd_info[["situation_id", "category_id", "product_id", "term_verbose_code"]]
		updated_weights = []
		for risk_id in situations:
			if risk_id not in updated_results:
				continue
			for cat_id in cat_ids:
				if cat_id not in updated_results[risk_id]:
					continue
				# print(risk_id, cat_id)
				df_aggr_info_related = df_aggr_info[(df_aggr_info["situation_id"] == risk_id) & (df_aggr_info["category_id"] == cat_id)]
				df_aggr_info_related = df_aggr_info_related[(df_aggr_info_related["product_id"] == df_aggr_info_related.iloc[0]["product_id"])]
				assert(len(df_aggr_info_related) == len(updated_results[risk_id][cat_id]["old_wts"]))
				assert(len(df_aggr_info_related) == len(updated_results[risk_id][cat_id]["new_wts"]))
				for t in range(len(df_aggr_info_related)):
					updated_weights.append({
						"situation_id": risk_id,
						"category_id": cat_id,
						"term_verbose_code": df_aggr_info_related.iloc[t]["term_verbose_code"],
						"old_weights": updated_results[risk_id][cat_id]["old_wts"][t],
						"new_weights": updated_results[risk_id][cat_id]["new_wts"][t],
					})
		df_updated_weights = pd.DataFrame(updated_weights)
		df_updated_weights.to_csv(outfile, index=False)
	'''
	Write updated weights results
	'''

	'''
	Compute NDCG with updated weights on test data
	'''
	with open(os.path.join(infolder, 'dataset_meta.txt')) as f:
		dataset_meta_data = f.read()
	dataset_meta = ast.literal_eval(dataset_meta_data)

	test_dataset = []
	lines = open(os.path.join(infolder, 'test_per_query_quid.txt')).readlines()
	for line in lines:
		attr = line.strip().split('\t')
		sid = int(attr[0].strip())
		qid = int(attr[1].strip())
		pids = json.loads(attr[2].strip())
		scores = json.loads(attr[3].strip())
		clicks = json.loads(attr[5].strip())
		test_dataset.append({
			'sid': sid,
			'qid': qid,
			'pids': pids,
			'scores': scores,
			'clicks': clicks
		})

	test_avg_ndcg_prev = 0
	test_avg_ndcg_updated = 0

	for test_item in test_dataset:
		test_meta = dataset_meta[test_item["qid"]]
		test_risk_set = test_meta[0]
		test_cat_id = test_meta[1]

		updated_scores = [0 for _ in range(max_list_size)]

		for test_risk in test_risk_set:
			if test_risk == 0:
				continue

			update_result_for_test = updated_results[test_risk][test_cat_id]

			for r, pid in enumerate(test_item["pids"]):
				if pid != 0:
					updated_scores[r] += update_result_for_test["rel_prd"][pid]["s_new"]
		
		# print(f"test id {test_item['qid']}: {updated_scores}")

		updated_pids_scores = [(test_item["pids"][r], updated_scores[r]) for r in range(max_list_size)]
		updated_pids_scores.sort(key=lambda a: a[1], reverse=True)
		# print(f"test id {test_item['qid']}: {updated_pids_scores}")

		updated_pids = [a[0] for a in updated_pids_scores]
		# print(f"test id {test_item['qid']}: {updated_pids}")

		prev_clicked_pids = []
		for r in range(max_list_size):
			if test_item["clicks"][r] == 1:
				prev_clicked_pids.append(test_item["pids"][r])

		updated_clicks = [0 for _ in range(max_list_size)]
		if len(prev_clicked_pids) > 0:
			for r in range(max_list_size):
				if updated_pids[r] in prev_clicked_pids:
					updated_clicks[r] = 1

		ndcg_prev = get_ndcg(test_item["clicks"])
		ndcg_updated = get_ndcg(updated_clicks)

		test_avg_ndcg_prev += ndcg_prev
		test_avg_ndcg_updated += ndcg_updated

		# print(f"test id {test_item['qid']}: {ndcg_prev} -> {ndcg_updated}")

	test_avg_ndcg_prev /= len(test_dataset)
	test_avg_ndcg_updated /= len(test_dataset)
	'''
	Compute NDCG with updated weights on test data
	'''

	print("updating rate", "=", updating_rate)
	print("previous NDCG", "=", test_avg_ndcg_prev)
	print("updated NDCG", "=", test_avg_ndcg_updated)

	# print(f"{updating_rate} {test_avg_ndcg_prev} {test_avg_ndcg_updated}")

if __name__ == '__main__':

	n_args = len(sys.argv)
	if n_args < 2:
		sys.exit("dataset folder not given")

	data_folder = os.path.join('.', sys.argv[1])

	if n_args >= 3:
		updating_rate = float(sys.argv[2])
	else:
		updating_rate = 0.1

	rewarding_scores, df_prd_info, df_cri_info = get_rewarding_scores(data_folder)

	run(updating_rate, rewarding_scores, df_prd_info, df_cri_info, data_folder)
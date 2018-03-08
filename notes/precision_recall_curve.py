import matplotlib.pyplot as plt 
def plot_precision_recall_curve(precision_1, recall_1,precision_2, recall_2, precision_3, recall_3):
	plt.plot(recall_1, precision_1, label = "System 1")
	plt.legend()
	plt.plot(recall_2, precision_2, label = "System 2")
	plt.legend()
	plt.plot(recall_3, precision_3, label = "System 3")
	plt.legend()
	plt.ylabel("Precision")
	plt.xlabel("Recall")
	plt.show()

precision_3 = [0.519031141869, 0.464912280702, 0.460641399417, 0.429333333333, 0.335416666667, 0.234463276836, 0.136150234742]
recall_3 = [0.625, 0.6625, 0.658333333333, 0.670833333333, 0.670833333333, 0.691666666667, 0.725]

precision_1 = [0.519031141869, 0.464912280702, 0.460641399417, 0.427027027027, 0.348017621145, 0.243647234679,  0.138245033113]
recall_1 = [0.625, 0.6625,0.658333333333,  0.658333333333, 0.658333333333,  0.679166666667,  0.695833333333]

precision_2 = [0.519031141869, 0.519031141869, 0.519031141869, 0.498392282958, 0.460410557185,0.365566037736, 0.212598425197]
recall_2 = [0.625,0.625,0.625,0.645833333333, 0.654166666667,0.645833333333,0.675]

# recall_1 = [0.625]
plot_precision_recall_curve(precision_1, recall_1,precision_2, recall_2,precision_3, recall_3)
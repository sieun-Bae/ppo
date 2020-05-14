import sys
import matplotlib.pyplot as plt 

def preprocess(lines):
	step = []
	reward = []

	line = lines[0].split(' ')
	step.append(int(line[4][:-1]))
	reward.append(float(line[-1]))

	lines = lines[1:]

	for line in lines:
		line = line.split(' ')
		step.append(int(line[3][:-1]))
		reward.append(float(line[-1]))

	return step, reward

def plot(step, reward):	
	plt.title("Reward per step")
	plt.plot(step, reward)
	plt.xlabel("step")
	plt.ylabel("reward")
	plt.show()

def main():
	file_name = sys.argv[1]

	f = open(f"./{file_name}", "r")
	lines = f.readlines()

	step, reward = preprocess(lines)
	plot(step, reward)

	f.close()

if __name__ == '__main__':
	main()
#"# of episode: {}, avg score: {:.1f}\n".format(n_epi, score/print_interval)
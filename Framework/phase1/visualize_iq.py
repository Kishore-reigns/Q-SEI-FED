import matplotlib.pyplot as plt

def plot_iq(iq_sample, title="IQ Signal"):
    I = iq_sample[:,0]
    Q = iq_sample[:,1]

    plt.figure(figsize=(10,4))
    plt.plot(I, color='blue')
    plt.plot(Q, color='blue')
    plt.legend()
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.show()

def plot_constellation(iq_sample):
    plt.figure(figsize=(5,5))
    plt.scatter(iq_sample[:,0], iq_sample[:,1], s=2)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title("IQ Constellation")
    plt.grid()
    plt.show()

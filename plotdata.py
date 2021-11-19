import matplotlib.pyplot as plt
import json

data = open('Modelo-90Acc.json','r')
data = json.load(data)

plt.plot(data['epochs'],data['train_loss'])
plt.show()

plt.plot(data['epochs'],data['train_accuracy'])
plt.show()

plt.plot(data['epochs'],data['test_loss'])
plt.show()

plt.plot(data['epochs'],data['test_accuracy'])
plt.show()
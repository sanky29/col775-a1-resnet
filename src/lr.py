import matplotlib.pyplot as plt 

'''
Assume lr is 1 in the start
2. ExponentialLR: (gamms = 0.95)
    1, 1*(0.95), 1*(0.95)^2, 1*(0.95)^3, 1*(0.95)^4, 1*(0.95)^5
3. LinearLR: (iters = 5, start_factor = 0.5, end_factor = 1)
    1*0.5, 1*0.6, 1*0.7, 1*0.8, 1*0.9, 1, 1, 1
4. StepLR: (step_size = 5, factor = 0.1)
    1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01
'''

lr_exp = [0.1]
for i in range(99):
    lr_exp.append(lr_exp[-1]*0.99)

lr_lin = []
lr = 0.1
sf = 0.5
ef = 1
iters = 50
m = (ef - sf)/iters
for i in range(100):
    if(i <= iters):
        lr_lin.append(lr*(sf + m*(i)))
    else:
        lr_lin.append(lr)

lr_step = []
lr = 0.1
step_size = 20
factor = 0.8
for i in range(100):
    e = i // step_size
    lr_step.append(lr*(factor**e))

plt.plot(lr_exp)
plt.plot(lr_lin)
plt.plot(lr_step)
plt.legend(['exp','lin','step'])
plt.xlabel('epoch')
plt.ylabel('lr')
plt.title('lr vs epoch')
plt.savefig('plots/lr.png')
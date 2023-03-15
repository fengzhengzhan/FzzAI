import os
for i in range(1, 11):
    path = 'E:\\1_mycode\\Knight_DQN\\getlabel\\knight'+str(i)+'.json'
    os.system("labelme_json_to_dataset "+path)
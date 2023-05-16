from dependencies import *


class AlgorithmReplayMemory:
    def __init__(self, table_name, path_autosave=projectpath.dateset_autosave_path):
        # 创建数据库文件夹
        path_sqlitedb = os.path.join(path_autosave, confglobal.DATASET_SQLITE)
        if not os.path.exists(path_sqlitedb):
            os.makedirs(path_sqlitedb)
            projlog(INFO, "[+] " + str(path_sqlitedb))

        # 创建数据库文件
        self.conn = sqlite3.connect(os.path.join(path_sqlitedb, confglobal.DATASET_SQLITE_FILENAME))
        projlog(INFO, "Open sqlite database successfully. " + str(self.conn))
        self.cursor = self.conn.cursor()
        # projlog(DEBUG, "Cursor object: " + str(self.cursor))

        self.table_name = table_name
        self.__createTable()

    def __del__(self):
        if self.conn is not None:
            self.conn.close()

    def __createTable(self):
        """
        table_name为 游戏名Table
        Parameters:
            id INT PRIMARY KEY NOT NULL  表示数量，主键自动增长
            time VARCHAR(32) NOT NULL  数据存入的时间
            state MEDIUMBLOB NOT NULL
            action INT NOT NULL
            allaction INT NOT NULL
            reward DOUBLE NOT NULL
            nextstate MEDIUMBLOB NOT NULL
            networkname VARCHAR(64) NOT NULL  网络名称
            epoch INT  轮数
            step INT  轮数中的步数
            loss DOUBLE  训练产生的loss值
            trainnum INT  用于训练的次数
            source VARCHAR(32)  数据来源，网络自动生成、人工手动操作获取
            manualscore TINYINT  是否进行人工打分，人工打分越高，说明操作越好，越需要被多次训练
            reservedword VARCHAR(64)  保留字段
        """
        # 查看所有表
        self.cursor.execute("SELECT name FROM sqlite_master WHERE TYPE='table' ORDER BY name")
        array_tables = self.cursor.fetchall()
        projlog(DEBUG, "Database tables: " + str(array_tables))

        if self.table_name not in array_tables:
            # 表字段
            self.cursor.execute(
                f"CREATE TABLE {self.table_name}("
                f"id INT PRIMARY KEY NOT NULL,"
                f"time VARCHAR(32) NOT NULL,"
                f"state MEDIUMBLOB NOT NULL,"
                f"action INT NOT NULL,"
                f"allaction INT NOT NULL,"
                f"reward DOUBLE NOT NULL,"
                f"nextstate MEDIUMBLOB NOT NULL,"
                f"networkname VARCHAR(64) NOT NULL,"
                f"epoch INT,"
                f"step INT,"
                f"loss DOUBLE,"
                f"trainnum INT NOT NULL,"
                f"source VARCHAR(32) NOT NULL,"
                f"manualscore TINYINT NOT NULL,"
                f"reservedword VARCHAR(64))")
            self.conn.commit()
            projlog(INFO, "[+] Create Table: " + str(self.table_name))

    # todo 将数据保存在内存中（按照数量大小设置上限），同时开启进程实时存储数据
    def insertData(self, time, state, action, allaction, reward, nextstate,
                   networkname, epoch, step, loss, trainnum, source, manualscore, reservedword):
        self.cursor.execute(
            f"INSERT INTO {self.table_name} "
            f"(time, state, action, allaction, reward, nextstate, "
            f"networkname, epoch, step, loss, trainnum, source, manualscore, reservedword, reservedword2) "
            f"VALUES ({time}, {state}, {action}, {allaction}, {reward}, {nextstate}, "
            f"{networkname}, {epoch}, {step}, {loss}, {trainnum}, {source}, {manualscore}, {reservedword})")

    def selectData(self):
        pass

    def updateData(self):
        pass

    def deleteData(self):
        pass


#     # 列表存储
#     def store_data(self, station, action, reward, next_station):
#         one_hot_action = np.zeros(self.action_size)
#         one_hot_action[action] = 1
#         self.replay_buffer.append((station, one_hot_action, reward, next_station))
#         if len(self.replay_buffer) > REPLAY_SIZE:
#             self.replay_buffer.pop(0)
#
#
#
#
# # 将REPLAY_BUFFER经验回放的存储库弹出一部分
# REPLAY_BUFFER = []
# if os.path.exists(DQN_STORE_PATH):
#     REPLAY_BUFFER = pickle.load(open(DQN_STORE_PATH, 'rb'))
# for i in range(600):
#     REPLAY_BUFFER.pop(len(REPLAY_BUFFER) - 1)
# pickle.dump(REPLAY_BUFFER, open(DQN_STORE_PATH, 'wb'))
# print(REPLAY_BUFFER, type(REPLAY_BUFFER), len(REPLAY_BUFFER))

if __name__ == '__main__':
    AlgorithmReplayMemory(confhk.GAME_NAME + confglobal.DATASET_DBTABLE_SUFFIX)

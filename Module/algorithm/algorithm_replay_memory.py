from dependencies import *


class AlgorithmReplayMemory:
    def __init__(self, table_name, path_autosave=projectpath.dateset_autosave_path):
        # 创建数据库文件夹
        path_sqlitedb = os.path.join(path_autosave, confglobal.DATASET_SQLITE)
        if not os.path.exists(path_sqlitedb):
            os.makedirs(path_sqlitedb)
            projlog(INFO, "[+] " + str(path_sqlitedb))

        # 创建数据库文件、
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
        # 查看所有表
        self.cursor.execute("SELECT name FROM sqlite_master WHERE TYPE='table' ORDER BY name")
        array_tables = self.cursor.fetchall()
        projlog(DEBUG, "Database tables: " + str(array_tables))
        # table_name为 游戏名Table
        if self.table_name not in array_tables:
            # 表字段
            # id,time,state,action,allaction,reward,nextstate,epoch,step,manualscore(0-10),reservedword,reservedword2
            self.cursor.execute(
                "CREATE TABLE {}("
                "id INT PRIMARY KEY NOT NULL,"
                "time VARCHAR(32) NOT NULL,"
                "state MEDIUMBLOB NOT NULL,"
                "action INT NOT NULL,"
                "allaction INT NOT NULL,"
                "reward DOUBLE NOT NULL,"
                "nextstate MEDIUMBLOB NOT NULL,"
                "epoch INT,"
                "step INT,"
                "manualscore TINYINT,"
                "reservedword VARCHAR(64),"
                "reservedword2 VARCHAR(64))".format(self.table_name))
            self.conn.commit()
            projlog(INFO, "[+] Create Table: " + str(self.table_name))

    def insertData(self, time, state, action, allaction, reward, nextstate, epoch, step, manualscore, reservedword,
                   reservedword2):
        self.cursor.execute(
            "INSERT INTO {} "
            "(time, state, action, totalaction, reward, nextstate, epoch, step, manualscore, reservedword,reservedword2) "
            "VALUES ({},{},{},{},{},{},{},{},{},{},{})"
            .format(self.table_name, time, state, action, allaction, reward, nextstate, epoch, step, manualscore,
                    reservedword, reservedword2))

    def selectData(self):
        pass

    def updateData(self):
        pass

    def deleteData(self):
        pass


if __name__ == '__main__':
    AlgorithmReplayMemory(confhk.GAME_NAME + confglobal.DATASET_DBTABLE_SUFFIX)

import sqlite3,time
import requests,subprocess
import json,udplib
f=open("/home/pi/share/config",'r')
test_string = f.read()
f.close()
res = json.loads(test_string)
host_ip=res['host_ip']
port_no=res['port_no']
# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
def select_task_by_priority(conn, priority):
    """
    Query tasks by priority
    :param conn: the Connection object
    :param priority:
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM temp_attendance WHERE ID=?", (priority,))

    rows = cur.fetchall()
    
    print(rows[0])
    return rows[0]
def delete_task(conn, id):
    """
        Delete a task by task id
        :param conn:  Connection to the SQLite database
        :param id: id of the task
        :return:        """
    sql = 'DELETE FROM temp_attendance WHERE ID=?'
    cur = conn.cursor()
    cur.execute(sql, (id,))
    conn.commit()
while 1:
    conn = sqlite3.connect('/home/pi/share/attendance.db')
    c = conn.cursor()
    c.execute("SELECT * FROM temp_attendance ORDER BY ID ASC LIMIT 1")
    details = c.fetchone()
    if(details !=None):
        print(details)
#         try:
        k=1
        while k:
            k=0
            ids=details[0]
            attendance=details[1]
            temperature=details[2]
            data=attendance+"~"+str(temperature)
            print(data)
            print(udplib.Attend_send(data,host_ip=host_ip,port_no=1111,bufferSize = 1024))
#             subprocess.Popen("echo 'attendance updated' | festival --tts",shell=True)
# #                     time.sleep(0.2)
            delete_task(conn,ids)
            time.sleep(2)
#         except:
#             print("exc")
#             pass
    #         delete_task(conn,row[0])
    conn.close()

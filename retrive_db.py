import sqlite3,time
import requests
try:
    import httplib
except:
    import http.client as httplib

def checkInternetHttplib(url="www.google.com", timeout=3):
    conn = httplib.HTTPConnection(url, timeout=timeout)
    try:
        conn.request("HEAD", "/")
        conn.close()
        return True
    except Exception as e:
#         print(e)
        return False
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
    if(checkInternetHttplib()):
#         try:
        for row in c.execute('SELECT ID FROM temp_attendance'):
                print(row[0])
                details=select_task_by_priority(conn,row[0])
                attendance=details[1]
                temperature=details[2]
                DeviceId=details[3]
                Date_time=details[4]
                print(attendance,temperature)
                link='http://skylogapi.coitor.com/api/attendance?employee_id='+str(attendance)+'&device_id='+str(DeviceId)+'&temprature='+str(temperature)+'&datetimed='+str(Date_time)
                print(link)
                print(requests.post(link).text)
                
                time.sleep(0.2)
                delete_task(conn,row[0])
#         except:
#             pass
    #         delete_task(conn,row[0])
    conn.close()

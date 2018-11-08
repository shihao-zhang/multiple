# the actions can be any kinds of control to building
import requests
from requests.auth import HTTPBasicAuth
import json
import base64
import pytz, datetime
import time
import ast


# Please change this into your own actuation system
class openHab():
    ##ref : https://github.com/openhab/openhab1-addons/wiki/Samples-REST#python
    """
    Actuation through openHab

    Parameters
    ----------
    host: str
    port: int
    username: str
    password: str
    """

    def __init__(self, host, port, username, password):  
        self.openhab_host = host
        self.openhab_port = port
        self.username = username
        self.password = password


    def basic_header(self):
        """ Header for OpenHAB REST request - standard """
        self.auth = base64.encodestring(('%s:%s' % (self.username, self.password)).encode()).decode().replace('\n', '')
        return {
            "Authorization" : "Basic %s" %self.auth,
            "Content-type": "text/plain"}
        
    
    def post_command(self, key, value):
        """ Post a command to OpenHAB 
        Parameters
        ----------
        key: str, item in the openhAB
        value: str, command from openHAB, e.g.,"ON" """

        url = 'http://%s:%s/rest/items/%s'%(self.openhab_host,
                                self.openhab_port, key)
        req = requests.post(url, data=value,
                                headers=self.basic_header())

        if req.status_code != requests.codes.ok:
            req.raise_for_status()



class plugWise():
    # http://128.2.108.76:8080/api/data.xml?type=appliances  ##check id and name mapping
    """
    Actuation through plugWise

    Parameters
    ----------
    host: str
    port: int

    """

    def __init__(self, host, port):  
        self.plugwise_host = host
        self.plugwise_port = port
    

    def send_command(self, key, value):
        """ send a command/action to plugwise - 
        Parameters
        ----------
        key: int, plugwiseid 
        value: str, action send to plugwise, e.g.,"switchoff" """

        url = 'http://%s:%s/api/actions.html?option=%s&id=%s'%(self.plugwise_host,
                                self.plugwise_port, value, key)
        payload = {'type': 'json'}
        req = requests.get(url, params=payload)

        if req.status_code != requests.codes.ok:
            req.raise_for_status()


class PIServer():
    """
    Data from PI server

    Parameters
    ----------
    host: str
    port: int
    username: str
    password: str
    """

    def __init__(self, host, username, password):  
        self.piserver_host = host
        self.headers = {'Content-type': 'application/json'} 
        self.auth = HTTPBasicAuth(username, password)
    

    def write_command(self, key):
        """ get data from plugwise - 
        Parameters
        ----------
        key: int, plugwiseid 
        value: str, action send to plugwise, e.g.,"switchoff" """

        url = 'https://%s/piwebapi/streams/%s/value?updateOption=replace'%(self.piserver_host, key)


        current_time = datetime.datetime.now() 
        timeStamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
        #print time 

        value = 1
        insertData = {
          "Timestamp": timeStamp,
          "UnitsAbbreviation": "C",
          "Good": True,
          "Questionable": False,
          "Value": value 
        }

        print(url)
        response = requests.post(url, data=json.dumps(insertData), headers=self.headers, auth=self.auth)
        print(response)

ACTION_Central = {
    "setpoint": "P0-MYhSMORGkyGTe9bdohw0AVzMCAAV0lOLTYyTlBVMkJWTDIwXElXX05PREUwMl9JVy5DT05UUk9MLlNFVFBPSU5U", 
}


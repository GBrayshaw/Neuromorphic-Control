import socket, struct, time
from threading import Thread, Lock

class ATI:
    def __init__(self, ip='192.168.1.1', port=49152):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.connect((ip, port))
        self._lock = Lock()

        self.mean = [0] * 6
        self.stream = False
        self.log = False
        self.data = None
        self.logged_data = {
            "force_x": [], "force_y": [], "force_z": [],
            "torque_x": [], "torque_y": [], "torque_z": []
        }

        self.tare(n=100)
        self.startStreaming()
        time.sleep(1)

    def send(self, command, count = 0):
        header = 0x1234
        message = struct.pack('!HHI', header, command, count)
        self.sock.send(message)

    def receive(self):
        rawdata = self.sock.recv(1024)

        if len(rawdata) < 36:
            raise RuntimeError(f"Short packet: {len(rawdata)} bytes")

        data = struct.unpack('!IIIiiiiii', rawdata[:36])[3:]
        with self._lock:
            self.data = [data[i] - self.mean[i] for i in range(6)]

            if self.log:
                fx, fy, fz, tx, ty, tz = self.to_newtons(self.data)
                keys = [
                    "force_x", "force_y", "force_z",
                    "torque_x", "torque_y", "torque_z"
                ]
                forces = [fx, fy, fz, tx, ty, tz]

                for key, force in zip(keys, forces):
                    self.logged_data[key].append(force)


    def to_newtons(self, data):
        force_x = data[0]/1000000
        force_y = -1*(data[1]/1000000)
        force_z = -1*(data[2]/1000000)

        torque_x = (data[3]/1000000)*1e3
        torque_y = (data[4]/1000000)*1e3
        torque_z = (data[5]/1000000)*1e3

        return force_x, force_y, force_z, torque_x, torque_y, torque_z

    def return_newtons(self):
        with self._lock:
            if self.data is None:
                return None
            else:
                return self.to_newtons(self.data)

    def new_sample(self):
        with self._lock:
            self.logged_data = {
                "force_x": [], "force_y": [], "force_z": [],
                "torque_x": [], "torque_y": [], "torque_z": []
            }

    def start_log(self):
        with self._lock:
            self.log = True

    def stop_log(self):
        with self._lock:
            self.log = False

    def return_log(self):
        with self._lock:
            return self.logged_data

    def return_log_avg(self):
        with self._lock:
            n = len(self.logged_data["force_x"])
            if n == 0:
                return None
            return {
                "force_x": sum(self.logged_data["force_x"]) / n,
                "force_y": sum(self.logged_data["force_y"]) / n,
                "force_z": sum(self.logged_data["force_z"]) / n,
                "torque_x": sum(self.logged_data["torque_x"]) / n,
                "torque_y": sum(self.logged_data["torque_y"]) / n,
                "torque_z": sum(self.logged_data["torque_z"]) / n,
            }

    def tare(self, n = 10):
        self.mean = [0] * 6
        self.getMeasurements(n = n)
        mean = [0] * 6
        for i in range(n):
            self.receive()
            for i in range(6):
                mean[i] += self.measurement()[i] / float(n)
        self.mean = mean
        return mean

    def receiveHandler(self):
        while self.stream:
            self.receive()

    def startStreaming(self, handler = True):
        self.getMeasurements(0)
        if handler:
            self.stream = True
            self.thread = Thread(target = self.receiveHandler)
            self.thread.daemon = True
            self.thread.start()
            print('FT Streaming')

    def getMeasurements(self, n):
        '''Args:
            n (int): The number of samples to request.
        '''
        self.send(2, count = n)

    def measurement(self):

        return self.data

    def stopStreaming(self):
        self.stream = False
        self.thread.join()
        time.sleep(0.1)
        self.send(0)
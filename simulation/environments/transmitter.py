class Transmitter():
    def __init__(self, name, lon, lat, span_lon=1.0, span_lat=1.0, height=10):
        self.name = name
        self.lon = lon
        self.lat = lat
        self.span_lon = span_lon
        self.span_lat = span_lat
        self.height = height

    def __call__(self):
        return (self.name, self.lon, self.lat, self.span_lon, self.span_lat, self.height)


transmitters = [
    Transmitter("MiReuk", 127.0411,  36.0317), #GUKAK
    Transmitter("MuDeung", 127.0033, 35.1283), #MBC
    Transmitter("SikJang", 127.4908, 36.3281), #MBC
    Transmitter("NamSan", 126.9892, 37.5522), #MBC
]

gist_transmitter = Transmitter("GIST", 126.8397, 35.2283)


test_transmitters = [
    Transmitter("ChungJu", 127.4997,  36.6430), #MBC
    Transmitter("Daegu", 128.4408, 35.8333), #DASAN MBC
    gist_transmitter
]


all_transmitters = transmitters + test_transmitters

print("-----------------TRAIN_TRANSMITTER LISTS-----------------")
for transmitter in transmitters:
    print(transmitter())
print("-----------------TEST_TRANSMITTER LISTS------------------")
for transmitter in test_transmitters:
    print(transmitter())
print("---------------------------------------------------------")

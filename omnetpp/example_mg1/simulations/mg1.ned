package example_mg1.simulations;

import example_mg1.Source;
import example_mg1.Server;
import example_mg1.Sink;

network mg1
{
    submodules:
        source: Source {
            @display("p=70,100");
        }
        server: Server {
            @display("p=200,100");
        }
        sink: Sink {
            @display("p=330,100");
        }
    connections:
        server.out --> sink.in;
        source.out --> server.in;
}


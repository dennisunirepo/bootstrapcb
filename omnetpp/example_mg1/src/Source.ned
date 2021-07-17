package example_mg1;

// Generate iid arrival events according to a distribution
simple Source
{
    parameters:
        volatile double interArrivalTime @unit(s);
        @display("i=block/source");
    gates:
        output out;
}
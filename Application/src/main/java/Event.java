
public class Event {
    private String type;
    private double value;
    private int count;

    public Event(String type, double value, int count) {
        this.type = type;
        this.value = value;
        this.count = count;
    }

    public String getType() {
        return type;
    }
    public int getCount() {
        return count;
    }
    public double getValue() {
        return value;
    }
}

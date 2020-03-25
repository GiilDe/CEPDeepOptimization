
public class Event {
    private String type;
    private double value;
    private int count;
    private int id;

    public Event(String type, double value, int count, int id) {
        this.type = type;
        this.value = value;
        this.count = count;
        this.id = id;
    }

    public String getType() {
        return type;
    }
    public int getId() { return id; }
    public double getValue() {
        return value;
    }
    public int getCount() { return count; }
    public void setCount(int count) { this.count = count; }

}

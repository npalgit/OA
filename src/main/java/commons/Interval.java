package commons;

/**
 * Created by tao on 8/21/17.
 */
public class Interval {
    public int start;
    public int end;
    public Interval(int s,int e){
        this.start=s;
        this.end=e;
    }

    public String toString(){
        return "["+start+" ,"+end+"]";
    }
}

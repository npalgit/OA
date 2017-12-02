package commons;

/**
 * Created by tao on 10/3/17.
 */
public class Point {
    public int x;
    public int y;
    public Point(int x,int y){
        this.x =x;
        this.y = y;
    }

    public String toString(){
        return "x: "+x+" y: "+y;
    }
}

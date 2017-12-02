package design.othello;

/**
 * Created by tao on 10/24/17.
 */
public class Location {
    private int row;
    private int col;

    public Location(int r,int c){
        row =r;
        col = c;
    }

    public boolean isSameAs(int r,int c){
        return col ==c && row ==r;
    }

    public int getRow(){
        return row;
    }
    public int getCol(){
        return col;
    }
}

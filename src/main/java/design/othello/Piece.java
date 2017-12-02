package design.othello;

/**
 * Created by tao on 10/24/17.
 */
public class Piece {
    private Color color;
    public Piece(Color c){
        this.color = c;
    }

    public void flip(){
        if(color == Color.Black)
            color = Color.White;
        else
            color = Color.Black;
    }

    public Color getColor(){
        return color;
    }

}

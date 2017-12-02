package design.othello;

/**
 * Created by tao on 10/24/17.
 */
public class Player {
    private Color color;
    public Player(Color c){
        color = c;
    }

    public int getScore(){
        return Game.getInstance().getBoard().getScoreForColor(color);
    }
    public boolean playPiece(int row,int col){
        return Game.getInstance().getBoard().placeColor(row,col,color);
    }

    public Color getColor(){
        return color;
    }
}

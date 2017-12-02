package design.othello;

/**
 * Created by tao on 10/24/17.
 */
public class Game {

    private Player[]players;
    private static Game instance;
    private Board board;
    private final int ROWS =10;
    private final int COLUMNS = 10;

    private Game(){
        //board = new Board();
    }

    public static Game getInstance(){
        if(instance == null)
            instance = new Game();
        return instance;
    }

    public Board getBoard(){
        return board;
    }
}

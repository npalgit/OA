package design.mine;

import java.util.Scanner;

/**
 * Created by tao on 10/24/17.
 */
public class Game {
    private Board board;
    private int rows;
    private int cols;
    private int bombs;
    private GameState state;

    public Game(int r,int c,int b){
        this.rows = r;
        this.cols = c;
        this.bombs = b;
        this.state = GameState.RUNNING;
    }

    public boolean initialize(){
        if(board ==null){
            board = new Board(rows,cols,bombs);
            board.printBoard(true);
            return true;
        }else{
            System.out.println("Game has been initialized");
            return false;
        }
    }

    public boolean start(){
        if(board==null){
            initialize();
        }
        return playGame();
    }

    public void printGameState(){
        if(state == GameState.LOST){
            board.printBoard(true);
            System.out.println("FAIL");
        }else if(state == GameState.WON){
            board.printBoard(true);
            System.out.println("WIN");
        }else{
            System.out.println("Number remaining: "+board.getNumRemaining());
        }
    }

    public boolean playGame(){
        Scanner scanner = new Scanner(System.in);
        printGameState();
        while(state == GameState.RUNNING){
            String input = scanner.nextLine();
            if(input.equals("exit")){
                scanner.close();
                return false;
            }

            UserPlay play = UserPlay.fromString(input);
            if(play==null){
                continue;
            }
            UserPlayResult result = board.playFlip(play);
            if(result.successfulMove()){
                state = result.getResultingState();
            }else{
                System.out.println("could not flip cell ("+play.getRow()+", "+ play.getCol()+") .");
            }
            printGameState();
        }
        scanner.close();
        return true;
    }


}

package design.othello;

import design.call.center.Director;

/**
 * Created by tao on 10/24/17.
 */
public class Board {

    private int blackCount = 0;
    private int whiteCount =0;
    private Piece[][] board;

    public Board(int rows,int cols){
        board = new Piece[rows][cols];
    }

    public void initialize(){
        int middleRow = board.length/2;
        int middleCol = board[middleRow].length/2;
        board[middleRow][middleCol] = new Piece(Color.White);
        board[middleRow+1][middleCol]=new Piece(Color.Black);
        board[middleRow+1][middleCol+1]= new Piece(Color.White);
        board[middleRow][middleCol+1]=new Piece(Color.Black);
        blackCount =2;
        whiteCount =2;

    }


    public boolean placeColor (int row, int col, Color color){
        if(board[row][col]!=null)
            return false;
        int []results = new int[4];
        results[0] = flipSection(row-1,col,color, Direction.up);
        results[1] = flipSection(row+1,col,color, Direction.down);
        results[2] = flipSection(row,col+1,color, Direction.right);
        results[3] = flipSection(row,col-1,color,Direction.left);

        int flipped = 0;
        for(int result:results){
            if(result>0){
                flipped +=result;
            }
        }

        if(flipped<0)
            return false;

        board[row][col] = new Piece(color);
        updateScore(color,flipped+1);
        return true;
    }


    private int flipSection(int row, int col, Color color, Direction d){
        int r = 0;
        int c =0;
        switch (d){
            case up:
                r =-1;
                break;
            case down:
                r=1;
                break;
            case left:
                c-=1;
                break;
            case right:
                c =1;
                break;
        }

        if(row<0||row>=board.length||col<0||col>=board[row].length||board[row][col]==null){
            return -1;
        }

        if(board[row][col].getColor()==color){
            return 0;
        }

        int flipped = flipSection(row+r,col+c,color,d);
        if(flipped<0)
            return -1;
        board[row][col].flip();
        return flipped+1;
    }

    public void updateScore(Color newColor, int newPiece){
        if(newColor == Color.Black){
            whiteCount -= newPiece -1;
            blackCount = newPiece;
        }else{
            blackCount -=newPiece-1;
            whiteCount +=newPiece;
        }
    }

    public int getScoreForColor(Color c){
        if(c==Color.Black)
            return blackCount;
        else
            return whiteCount;
    }

    public void printBoard(){
        for(int r =0; r<board.length;++r){
            for(int c=0;c<board[r].length;++c){
                if(board[r][c]==null){
                    System.out.println("_");
                }else if(board[r][c].getColor()==Color.White){
                    System.out.println("W");
                }else{
                    System.out.println("B");
                }
            }
            System.out.println();
        }

    }
}

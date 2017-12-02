package design.mine;

import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;

/**
 * Created by tao on 10/24/17.
 */

public class Board {

    private int nRows;
    private int nCols;
    private int nBombs = 0;
    private Cell[][] cells;
    private Cell[]bombs =null;
    private int numUnexposedRemaining;

    public Board(int r,int c,int b){
        nBombs = b;
        nRows = r;
        nCols= c;
        initializeBoard();
        shuffleBoard();
        setNumberedCells();
        numUnexposedRemaining = nRows*nCols-nBombs;
    }

    public void initializeBoard(){
        cells = new Cell[nRows][nCols];
        bombs = new Cell[nBombs];
        for(int r =0; r<nRows;++r){
            for(int c=0; c<nCols;++c){
                cells[r][c]=new Cell(r,c);
            }
        }

        for(int i=0;i<nBombs;++i){
            int r = i/nCols;
            int c = (i-r*nCols)%nCols;
            bombs[i] = cells[r][c];
            bombs[i].setBomb(true);
        }
    }

    public void shuffleBoard(){
        int nCells = nRows* nCols;
        Random random = new Random();
        for(int index1=0; index1<nCells;++index1){
            int index2 = index1+random.nextInt(nCells-index1);
            if(index1!=index2){
                int row1 = index1/nCols;
                int col1 = (index1- row1*nCols)%nCols;
                Cell cell1 = cells[row1][col1];

                int row2 = index2/nCols;
                int col2 = (index2-row2*nCols)%nCols;
                Cell cell2 = cells[row2][col2];

                cells[row1][col1]= cell2;
                cell2.setRowAndColumn(row1,col1);
                cells[row2][col2] = cell1;
                cell1.setRowAndColumn(row2,col2);

            }
        }
    }


    public void expandBlank( Cell cell){
        int [][]deltas={
                {-1,-1},{-1,0},{-1,1},
                {0,-1}, {0,1},
                {1,-1},{1,0},{1,1}
        };
        Queue<Cell> toExplore = new LinkedList<>();
        toExplore.add(cell);
        while(!toExplore.isEmpty()){
            Cell current = toExplore.poll();
            for(int[]delta: deltas){
                int r = current.getRow()+ delta[0];
                int c = current.getCol() + delta[1];
                if(inBound(r,c)){
                    Cell neighbor = cells[r][c];
                    if(flipCell(neighbor) && neighbor.isBlank()){
                        toExplore.add(neighbor);
                    }
                }
            }
        }
    }
    public UserPlayResult playFlip(UserPlay play){
        Cell  cell = getCellAtLocation(play);
        if(cell == null){
            return new UserPlayResult(false,GameState.RUNNING);
        }
        if(play.isGuess()){
            boolean guessResult = cell.toggleGuess();
            return new UserPlayResult(guessResult,GameState.RUNNING);
        }

        boolean result = flipCell(cell);
        if(cell.isBomb())
            return new UserPlayResult(result,GameState.LOST);
        if(cell.isBlank()){
            expandBlank(cell);
        }

        if(numUnexposedRemaining==0)
                return new UserPlayResult(result,GameState.WON);
        return new UserPlayResult(result,GameState.RUNNING);
    }


    public void printBoard(boolean showUnderside){
        System.out.println();
        System.out.print(" ");
        for(int i=0;i<nCols;++i){
            System.out.print(i+ " ");
        }

        System.out.println();
        for(int i=0;i<nCols;++i){
            System.out.print("--");
        }

        System.out.println();
        for(int r =0;r<nRows;++r){
            System.out.println(r+"| ");
            for(int c=0;c<nCols;++c){
                if(showUnderside){
                    System.out.println(cells[r][c].getUndersideState());
                }else
                    System.out.println(cells[r][c].getSurfaceState());
            }
            System.out.println();
        }
    }
    public boolean flipCell(Cell cell){
        if(!cell.isExposed() && !cell.isGuess()){
            cell.flip();
            numUnexposedRemaining--;
            return true;
        }
        return false;
    }

    public boolean inBound(int row, int col){
        return row>=0 && row<nRows && col>=0 && col<nCols;
    }

    public void setNumberedCells(){
        int [][]deltas={
                {-1,-1},{-1,0},{-1,1},
                {0,-1}, {0,1},
                {1,-1},{1,0},{1,1}
        };
        for(Cell bomb:bombs){
            int row = bomb.getRow();
            int col = bomb.getCol();
            for(int []delta:deltas){
                int r = row+delta[0];
                int c = col+ delta[1];
                if(inBound(r,c)){
                    cells[r][c].incrementNumber();
                }
            }
        }
    }

    public Cell getCellAtLocation(UserPlay play){
        int row = play.getRow();
        int col = play.getCol();
        if(!inBound(row,col))
            return null;
        return cells[row][col];
    }

    public int getNumRemaining(){
        return numUnexposedRemaining;
    }

}

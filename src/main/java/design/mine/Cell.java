package design.mine;

/**
 * Created by tao on 10/24/17.
 */
public class Cell {

    private int row;
    private int col;
    private boolean isBomb;
    private int number;
    private boolean isExposed = false;
    private boolean isGuess = false;

    public Cell(int r,int c){
        isBomb = false;
        number = 0;
        row =r;
        col = c;
    }

    public void setRowAndColumn(int r,int c){
        row = r;
        col = c;
    }

    public void setBomb(boolean bomb){
        isBomb = bomb;
        number = -1;
    }

    public void incrementNumber(){
        number++;
    }

    public int getRow(){
        return row;
    }

    public int getCol(){
        return col;
    }

    public boolean isBomb(){
        return isBomb;
    }

    public boolean isBlank(){
        return number==0;
    }

    public boolean isExposed(){
        return isExposed;
    }

    public boolean flip(){
        isExposed = true;
        return !isBomb;
    }

    public boolean toggleGuess(){
        if(!isExposed){
            isGuess = !isGuess;
        }
        return isGuess;
    }

    public boolean isGuess(){
        return isGuess;
    }

    public String toString(){
        return getUndersideState();
    }

    public String getUndersideState(){
        if(isBomb){
            return "* ";
        }else if(number>0){
            return Integer.toString(number)+" ";
        }else
            return " ";
    }

    public String getSurfaceState(){
        if(isExposed){
            return getUndersideState();
        }else if(isGuess){
            return "B ";
        }else
            return "? ";
    }


}


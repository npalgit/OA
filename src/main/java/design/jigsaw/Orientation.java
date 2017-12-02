package design.jigsaw;

/**
 * Created by tao on 10/24/17.
 */
public enum Orientation {
    LEFT, TOP, RIGHT, BOTTOM;
    public Orientation getOpposite(){
        switch (this){
            case LEFT: return RIGHT;
            case RIGHT: return LEFT;
            case TOP: return BOTTOM;
            case BOTTOM: return TOP;
            default: return null;
        }
    }
}

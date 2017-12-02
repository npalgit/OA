package design.jigsaw;

/**
 * Created by tao on 10/24/17.
 */
public enum  Shape {
    INNER, OUTER, FLAT;
    public Shape getOppsite(){
        switch (this){
            case INNER:
                return OUTER;
            case OUTER:
                return INNER;
            default:
                return null;
        }
    }
}

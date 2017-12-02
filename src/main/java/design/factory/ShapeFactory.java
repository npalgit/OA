package design.factory;

/**
 * Created by tao on 10/14/17.
 */
public class ShapeFactory {

    public Shape getShape(String shapeType){
        if(shapeType == null){
            return null;
        }
        if(shapeType.equals("CIRCLE"))
            return new Circle();
        else if(shapeType.equals("SQUARE"))
            return new Square();
        else if(shapeType.equalsIgnoreCase("RECTANGLE"))
            return new Rectangle();
        return null;
    }
}

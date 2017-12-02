package design.jigsaw;

/**
 * Created by tao on 10/24/17.
 */
public class Edge {
    private Shape shape;
    private String code;
    private Piece parentPiece;

    public Edge(Shape shape, String code){
        this.shape = shape;
        this.code = code;
    }

    public String getCode(){
        return code;
    }

    public Edge _createMatchingEdge(){
        if(shape == Shape.FLAT)
            return null;
        return new Edge(shape.getOppsite(),getCode());
    }

    public boolean fitsWith(Edge edge){
        return edge.getCode().equals(getCode());
    }

    public void setParentPiece(Piece parentPiece){
        this.parentPiece = parentPiece;
    }

    public Piece getParentPiece(){
        return parentPiece;
    }

    public Shape getShape(){
        return shape;
    }

    public String toString(){
        return code;
    }
}

package design.jigsaw;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by tao on 10/24/17.
 */
public class Piece {
    public final static int NUMBER_OF_EDGES = 4;
    private Map<Orientation, Edge> edges = new HashMap<>();
    public Piece(Edge[]edgeList){
        Orientation []orientations = Orientation.values();

        for(int i=0;i<edgeList.length;++i){
            Edge edge = edgeList[i];
            edge.setParentPiece(this);
            edges.put(orientations[i],edge);
        }
    }

    public void setEdgeAsOrientation(Edge edge, Orientation orientation){
        Orientation currentOrientation = getOrientation(edge);

    }

    public Orientation getOrientation(Edge edge){
        for(Map.Entry<Orientation, Edge>entry: edges.entrySet()){
            if(entry.getValue() == edge)
                return entry.getKey();
        }
        return null;
    }


    public void rotateEdgesBy(int numberRotations){
        Orientation []orientations = Orientation.values();
        Map<Orientation,Edge> rotated = new HashMap<>();

        numberRotations = numberRotations%NUMBER_OF_EDGES;
        if(numberRotations<0)
            numberRotations+=NUMBER_OF_EDGES;
        for(int i=0;i<orientations.length;++i){
            Orientation oldOrientation = orientations[(i-numberRotations+NUMBER_OF_EDGES)%NUMBER_OF_EDGES];
            Orientation newOrientation = orientations[i];
            rotated.put(newOrientation, edges.get(oldOrientation));
        }
        edges = rotated;
    }
    public boolean isCornner(){
        Orientation []orientations = Orientation.values();
        for(int i=0;i<orientations.length;++i){
            Shape current = edges.get(orientations[i]).getShape();
            Shape next = edges.get(orientations[(i+1)%NUMBER_OF_EDGES]).getShape();
            if(current == Shape.FLAT && next == Shape.FLAT)
                return true;
        }
        return false;
    }


    public boolean isBorder(){
        Orientation [] orientations = Orientation.values();
        for(int i=0;i<orientations.length;++i){
            if(edges.get(orientations[i]).getShape() == Shape.FLAT)
                return true;
        }
        return false;
    }

    public Edge getEdgeWithOrientation(Orientation orientation){
        return edges.get(orientation);
    }

    public Edge getMatchingEdge(Edge targetEdge){
        for(Edge e: edges.values()){
            if(targetEdge.fitsWith(e))
                return e;
        }
        return null;
    }

    public String toString(){
        StringBuilder sb = new StringBuilder();

        Orientation []orientations = Orientation.values();
        for(Orientation o: orientations){
            sb.append(edges.get(o.toString()+","));
        }
        return "["+sb.toString()+"]";
    }

}

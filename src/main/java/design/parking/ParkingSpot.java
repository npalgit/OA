package design.parking;

/**
 * Created by tao on 10/24/17.
 */
public class ParkingSpot {
    private Vehicle vehicle;
    private VehicleSize spotSize;
    private int row;
    private int spotNumber;
    private Level level;

    public ParkingSpot(Level level, int r,int n,VehicleSize s){
        this.level =level;
        this.row = r;
        this.spotNumber = n;
        this.spotSize = s;
    }

    public boolean isAvailable(){
        return vehicle ==null;
    }


    //check if the spot is big enough and is available
    public boolean canFitVehicle(Vehicle vehicle){
        return isAvailable() && vehicle.canFitInSpot(this);
    }


    // park vehicle in this spot
    public boolean park(Vehicle v){
        if(!canFitVehicle(v)){
            return false;
        }
        vehicle = v;
        vehicle.parkInSpot(this);
        return true;
    }

    public int getRow(){
        return row;
    }

    public int getSpotNumber(){
        return spotNumber;
    }

    public VehicleSize getSpotSize(){
        return spotSize;
    }
    //remove vehicle from spot, and notify level that a new spot is available
    public void removeVehicle(){
        level.spotFreed();
        vehicle = null;
    }
    public void print(){
        if(vehicle==null){
            if(spotSize == VehicleSize.Compact){
                System.out.println("C");
            }else if(spotSize == VehicleSize.Large){
                System.out.println("L");
            }else if(spotSize == VehicleSize.Motorcycle){
                System.out.println("m");
            }
        }else
            vehicle.print();
    }
}


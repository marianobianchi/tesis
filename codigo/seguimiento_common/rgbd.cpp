


std::pair<std::pair<float,float>,std::pair<float,float> >
from_flat_to_cloud_limits(std::pair<unsigned int, unsigned int> topleft,
                          std::pair<unsigned int, unsigned int> bottomright,
                          std::string depth_filename)
{
    /**
     * Dada una imagen en profundidad y la ubicación de un objeto
     * en coordenadas sobre la imagen en profundidad, obtengo la nube
     * de puntos correspondiente a esas coordenadas y me quedo
     * unicamente con los valores máximos y mínimos de las coordenadas
     * "x" e "y" de dicha nube
     * 
     * REVISAR: creo que para PCL "x" corresponde a las columnas e "y" a las filas
     **/

}
